# rag_engine/chain.py
from __future__ import annotations
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda
from rag_engine.prompt import get_search_prompt
from transformers import AutoTokenizer


from typing import Tuple, List, Dict
from pathlib import Path
import yaml

from rag_engine.retrievers.base import Query, Document
from rag_engine.retrievers.builder import build_components
from rag_engine.retrievers.fuse import fuse_weighted

CONFIG_PATH = Path("config.yaml")

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")

def build_search_chain(llm, retriever):
    prompt = get_search_prompt()


    def format_with_source(inputs):
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)

        # 전체 context 합치기
        full_context = "\n".join([doc.page_content for doc in docs])

        # context = "\n".join([doc.page_content for doc in docs])
        # sources = ", ".join(sorted(set([doc.metadata.get("source", "?") + f" (p{doc.metadata.get('page', '?')})" for doc in docs])))
        
        
        # 토큰 길이 제한 (예: 800토큰만 사용)
        tokens = tokenizer.tokenize(full_context)
        if len(tokens) > 800:
            tokens = tokens[:800]
        context = tokenizer.convert_tokens_to_string(tokens)

        sources = ", ".join(sorted(set(
            [doc.metadata.get("source", "?") + f" (p{doc.metadata.get('page', '?')})" for doc in docs]
        )))
        
        return {
            "question": question,
            "context": context,
            "sources": sources
        }

    chain = (
        RunnableLambda(format_with_source)
        | prompt
        | llm
        | RunnableLambda(lambda x: x + "\n\n 출처: " + x["sources"] if isinstance(x, dict) else x)
        | StrOutputParser()
    )
    return chain


def build_chat_chain(llm, retriever):
    prompt = get_search_prompt()

    def merge_chat_history(input):
        history = input.get("chat_history", [])
        chat_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in history])
        full_question = f"{chat_str}\nUser: {input['question']}"
        docs = retriever.get_relevant_documents(full_question)
        context = "\n".join([doc.page_content for doc in docs])
        sources = ", ".join(sorted(set([doc.metadata.get("source", "?") + f" (p{doc.metadata.get('page', '?')})" for doc in docs])))
        return {
            "question": full_question,
            "context": context,
            "sources": sources
        }

    chain = (
        RunnableLambda(merge_chat_history)
        | prompt
        | llm
        | RunnableLambda(lambda x: x + "\n\n 출처: " + x["sources"] if isinstance(x, dict) else x)
        | StrOutputParser()
    )
    return chain

def _load_cfg() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _format_context(docs: List[Document]) -> str:
    print("🔥🔥🔥 새로운 _format_context 함수가 실행되었습니다! 🔥🔥🔥") 
    """LLM에 투입할 컨텍스트 문자열 포맷"""
    return "\n\n".join([d.content for d in docs])



def get_context(query_text: str) -> Tuple[str, List[Document]]:
    """
    config.yaml 기반으로 모든 리트리버/리랭커를 만들고,
    일부 리트리버에서 에러가 발생하더라도 멈추지 않고,
    성공한 결과만으로 최종 컨텍스트를 반환하는 튼튼한 함수입니다.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    components = build_components(cfg)
    retrievers = components["retrievers"]
    reranker = components["reranker"]
    weights = components["weights"]

    results: Dict[str, List[Document]] = {}
    
    # --- [핵심] 모든 리트리버를 순회하며 예외 처리 ---이 문단 디비연결되면 살리기!!!!
    # for name, retriever in retrievers.items():
    #     try:
    #         config = cfg["retrievers"][name]
    #         k = config.get("n_results", 8)
    #         print(f"🔍 '{name}'에서 {k}개 검색...")
            
    #         docs = retriever.retrieve(Query(query_text, top_k=k))

    #         # 이름이 'pg'로 시작하는 리트리버 결과에만 리랭커 적용
    #         if name.startswith("pg") and reranker:
    #             print(f"🧐 '{name}' 결과 리랭킹...")
    #             rer_topk = cfg.get("reranker", {}).get("top_k")
    #             docs = reranker.rerank(query_text, docs, top_k=rer_topk)
            
    #         results[name] = docs
            
    #     except Exception as e:
    #         # ❗️ 에러가 발생해도 프로그램을 멈추지 않고, 경고 메시지만 출력합니다.
    #         print(f"🔥🔥🔥 경고: '{name}' 리트리버 실행 중 에러 발생! 이 리트리버를 건너뜁니다.")
    #         print(f"에러 원인: {e}")
    #         # results 딕셔너리에 아무것도 추가하지 않고 그냥 넘어갑니다.
    # --- [핵심] 모든 리트리버를 순회하며 예외 처리 ---
    for name, retriever in retrievers.items():
        # 🔽 [수정] 이름이 'pg'로 시작하면 이 리트리버는 건너뜁니다.
        if name.startswith("pg"):
            print(f"⚠️ [임시 비활성화] '{name}' 리트리버를 건너뜁니다.")
            continue

        try:
            config = cfg["retrievers"][name]
            k = config.get("n_results", 8)
            print(f"🔍 '{name}'에서 {k}개 검색...")

            docs = retriever.retrieve(Query(query_text, top_k=k))
            
            # pg 리트리버는 건너뛰므로, 이 리랭킹 로직은 실행되지 않습니다.
            if name.startswith("pg") and reranker:
                print(f"🧐 '{name}' 결과 리랭킹...")
                rer_topk = cfg.get("reranker", {}).get("top_k")
                docs = reranker.rerank(query_text, docs, top_k=rer_topk)

            results[name] = docs

        except Exception as e:
            # ❗️ 에러가 발생해도 프로그램을 멈추지 않고, 경고 메시지만 출력합니다.
            print(f"🔥🔥🔥 경고: '{name}' 리트리버 실행 중 에러 발생! 이 리트리버를 건너뜁니다.")
            print(f"에러 원인: {e}")
            
            
    # --- 에러 없이 성공한 결과들만으로 가중 병합 ---
    if not results:
        print("⚠️ 모든 리트리버에서 정보를 가져오지 못했습니다. 빈 컨텍스트를 반환합니다.")
        return "", []

    print("🔄 성공적으로 검색된 결과들을 가중 병합합니다...")
    final_top_k = cfg.get("final_top_k", 10)
    fused_docs = fuse_weighted(results, weights, top_k=final_top_k)

    context_str = _format_context(fused_docs)
    return context_str, fused_docs



# 선택) LLM 호출까지 묶는 헬퍼 (네가 원하는 llm_fn 시그니처에 맞춰 사용)
def build_answer(llm_fn, user_question: str, prompt_tmpl: str) -> Dict:
    """
    llm_fn: callable(prompt: str) -> str
    prompt_tmpl: 예) 'Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:'
    """
    context_str, docs = get_context(user_question)
    prompt = prompt_tmpl.format(context=context_str, question=user_question)
    answer = llm_fn(prompt)

    sources = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("filename") or d.id
        sources.append({"id": d.id, "source": src, "score": d.score})

    return {"answer": answer, "sources": sources}
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
    """LLM에 투입할 컨텍스트 문자열 포맷"""
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source") or d.metadata.get("filename") or d.id
        lines.append(f"[{i}] (source: {src})\n{d.content}\n")
    return "\n---\n".join(lines)

# def get_context(query_text: str) -> Tuple[str, List[Document]]:
#     """
#     멀티 리트리버로 문서 수집 → (옵션)리랭킹 → 가중 통합(3:7 등) → 컨텍스트 생성
#     반환: (context_str, fused_docs)
#     """
#     cfg = _load_cfg()
#     retrievers, reranker, weights, rer_topk = build_components(cfg)

#     results: Dict[str, List[Document]] = {}

#     if retrievers.get("chroma"):
#         k = cfg["retrievers"]["chroma"].get("n_results", 8)
#         results["chroma"] = retrievers["chroma"].retrieve(Query(query_text, top_k=k))

#     if retrievers.get("pg"):
#         k = cfg["retrievers"]["pg"].get("n_results", 8)
#         results["pg"] = retrievers["pg"].retrieve(Query(query_text, top_k=k))

#     # (옵션) 개별 결과 리랭킹
#     if reranker and rer_topk:
#         for name in list(results.keys()):
#             results[name] = reranker.rerank(query_text, results[name], top_k=rer_topk)

#     # 가중 병합 (config.yaml 의 weights 사용; 예: chroma 0.7, pg 0.3)
#     final_top_k = cfg.get("final_top_k", 10)
#     fused_docs = fuse_weighted(results, weights, top_k=final_top_k)

#     context_str = _format_context(fused_docs)
#     return context_str, fused_docs


def get_context(query_text: str) -> Tuple[str, List[Document]]:
    """
    config.yaml 파일을 읽어, 그 안에 정의된 모든 리트리버와 리랭커를
    자동으로 생성하고, PostgreSQL 결과에만 리랭커를 적용한 뒤,
    최종적으로 모든 정보를 가중 병합하여 컨텍스트를 반환합니다.
    """
    # 1. 설정 파일(설계도) 로드
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 설계도를 바탕으로 모든 부품(리트리버, 리랭커 등) 조립
    components = build_components(cfg)
    retrievers = components["retrievers"]
    reranker = components["reranker"]
    weights = components["weights"]

    # 3. 조립된 모든 리트리버를 사용해 각자 정보 수집
    results: Dict[str, List[Document]] = {}
    for name, retriever in retrievers.items():
        config = cfg["retrievers"][name]
        k = config.get("n_results", 8)
        print(f"🔍 '{name}' 리트리버에서 {k}개 문서를 검색합니다...")
        
        docs = retriever.retrieve(Query(query_text, top_k=k))

        # ❗️ [핵심] 이름이 'pg'로 시작하는 리트리버 결과에만 리랭커(검수 전문가) 적용
        if name.startswith("pg") and reranker:
            print(f"🧐 '{name}' 리트리버의 결과를 리랭킹합니다...")
            reranker_top_k = cfg.get("reranker", {}).get("top_k")
            docs = reranker.rerank(query_text, docs, top_k=reranker_top_k)
        
        results[name] = docs

    # 4. 수집된 모든 정보들을 가중치에 따라 종합
    print("🔄 모든 검색 결과를 가중 병합합니다...")
    final_top_k = cfg.get("final_top_k", 10)
    fused_docs = fuse_weighted(results, weights, top_k=final_top_k)

    # 5. LLM이 이해하기 쉬운 형식으로 최종 보고서(컨텍스트) 작성
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
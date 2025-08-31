from langchain.prompts import PromptTemplate
#from rag_engine.llm import load_llm
from rag_engine.embedder import get_embedder
from rag_engine.vector_store import load_vector_db, add_to_vector_db
from rag_engine.search import search_serper
from transformers import AutoTokenizer
from langchain.schema import Document
import uuid
import json
from typing import List, Optional
from fastapi import UploadFile, FastAPI, HTTPException, Header
from rag_engine.prompt import get_search_prompt
from rag_engine.loader import load_pdf
import requests
from fastapi import FastAPI, Form
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag_engine.process import process_multiple_files
from rag_engine.tagger import generate_keyword_tags
from rag_engine.captioner import generate_captions
import io, base64, uuid, os
from cli_runner import is_supported
from rag_engine.process import process_multiple_files



load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def call_claude(prompt: str, max_tokens: int = 512) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["content"][0]["text"]


def _is_supported(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith((".pdf", ".csv"))



def generate_report(
    topic: str,
    file: Optional[UploadFile] = None,
    file_paths: Optional[List[str]] = None,
    references: Optional[List[str]] = None,
):
    print("📌 1. 함수 진입 - topic:", topic)
    
    
        
    # 벡터DB 열고 검색
    embedder = get_embedder()
    vectordb = load_vector_db(embedder)
    
    

    # 4. 관련 문서 검색(내부검색)
    retriever = vectordb.as_retriever()
    internal_docs = retriever.get_relevant_documents(topic)
    
    # 외부 검색 (Serper)
    #external_docs = search_serper(topic, num_results=3)

    # 6. 문서 통합 및 컨텍스트 구성성
    # all_docs = internal_docs + external_docs
    all_docs = internal_docs
    context = "\n\n".join([doc.page_content for doc in all_docs])
    print("✅ 6. 문서 통합 완료 - 문서 수:", len(all_docs))

    # 5. references가 있다면 문맥 뒷부분에 사용자 요구사항으로 붙이기
    if references:
        requirements_text = "\n\n[사용자 요구사항]\n" + "\n".join(references)
        context += requirements_text

    prompt_template = get_search_prompt() #-> completion호출 방식때 겟서치 프롬프트 함수
    full_prompt = prompt_template.format(context=context, question=topic)
    print("📌 7. 프롬프트 생성 완료")

    
    # ✅ Claude API로 기사 본문 생성
    try:
        output = call_claude(full_prompt, max_tokens=8000)
        print("💬 Claude 응답 도착")
    except Exception as e:
        import traceback
        print("❌ Claude API 호출 실패:")
        traceback.print_exc()
        output = None


    if output:
        title_prompt = f"""
        당신은 스포츠 기사 제목 생성 전문가입니다.

        아래 기사 내용을 바탕으로 **적절하고 임팩트 있는 기사 제목 3개**를 제안하세요. 
        조건:
        - 각 제목은 한 줄, 15~20자 이내 권장
        - 번호를 매기지 말고, JSON 배열 형식으로 반환하세요.
        
        [출력 형식]
        ["제목1", "제목2", "제목3"]

        기사 내용:
        {output.strip()}
        """
        try:
            raw = call_claude(title_prompt, max_tokens=128).strip()
            import json, re
            try:
                titles = json.loads(raw)
                assert isinstance(titles, list), "titles is not a list"
            except Exception:
                # 폴백: 따옴표 안 문자열들 or 줄 단위 3개
                candidates = re.findall(r'"([^"]{5,40})"', raw) or \
                             [s.strip() for s in raw.splitlines() if s.strip()]
                titles = candidates[:3]
            # 클린업 & 개수 보정
            titles = [t.strip(' "\'“”') for t in titles if t.strip()][:3]
            if len(titles) == 0:
                titles = ["기사 제목 생성 실패"]
            print("📝 기사 제목 3개 생성:", titles)
        except Exception as e:
            titles = ["기사 제목 생성 실패"]
            print("❌ 제목 생성 오류:", e)
            
            
    #  태그 생성 (LLM 콜백으로 call_claude 주입)
    if output:
        try:
            print("🎯 태그 생성 시작 - article 길이:", len(output), "topic:", topic)
            tags = generate_keyword_tags(
                article=output,
                topic=topic,
                llm_fn=lambda p, mt: call_claude(p, max_tokens=mt),  # ← 의존성 주입
                max_tags=12
            )
            print("🏷️ 태그 생성 완료:", tags, "(타입:", type(tags), "길이:", len(tags), ")")
        except Exception as e:
            print("❌ 태그 생성 실패:", e)
            import traceback
            traceback.print_exc()
            tags = []
    else:
        print("❌ output이 없어서 태그 생성 건너뜀")
        tags = []
        
    # 트윗생성
    if output:
        try:
            captions = generate_captions(
                article=output,
                topic=topic,
                llm_fn=lambda p, mt: call_claude(p, max_tokens=mt),
           )
            print("🪄 캡션 생성 완료:", captions)
        except Exception as e:
            print("❌ 캡션 생성 실패:", e)
            captions = {}
    else:
        captions = {}


    # 9. 출처 수집
    sources = []
    for doc in all_docs:
        src = doc.metadata.get("source")
        if src and src not in sources:
            sources.append(src)


    # 10. JSON 형태로 반환
    result = {
        "user_request": f"{topic}",
        "title": titles, 
        "content": output.strip(),
        "sources": sources,
        "tags": tags,
        "captions": captions,
    }
    print("📦 최종 반환 결과:")
    print(f"  - title: {result['title']} (개수: {len(result['title'])})")
    print(f"  - content 길이: {len(result['content'])}")
    print(f"  - sources: {result['sources']}")
    print(f"  - tags: {result['tags']} (타입: {type(result['tags'])}, 길이: {len(result['tags'])})")
    print(f"  - captions: {result['captions']}")
    return result

def collect_file_paths(raw: str):
    """입력받은 경로나 문자열에서 유효한 PDF/CSV 파일 경로 리스트를 반환"""
    if os.path.isdir(raw):
        file_paths = [
            os.path.join(raw, f)
            for f in os.listdir(raw)
            if f.lower().endswith((".pdf", ".csv"))
        ]
    else:
        file_paths = [
            p.strip() for p in raw.split(",") if is_supported(p.strip())
        ]
    return file_paths



if __name__ == "__main__":
    print(" 처리할 PDF 또는 CSV 파일 경로를 입력하세요.")
    print(" - 쉼표(,)로 구분해서 여러 개 가능")
    print(" - 또는 폴더 경로 입력 시 하위 PDF/CSV 모두 처리됨")
    raw = input("입력: ").strip()

    file_paths = collect_file_paths(raw)

    if not file_paths:
        print(" PDF 또는 CSV 파일이 없습니다.")
    else:
        total_chunks = process_multiple_files(file_paths)
        print(f"\n 총 {len(file_paths)}개 파일, {total_chunks}개의 청크가 저장되었습니다.")

    
    topic = input(" 기사 주제를 입력하세요: ")
    report = generate_report(topic, file=None, references=None)
    print("\n📄 기사 요구사항:")
    print(report["user_request"])
    print("\n📰 제목 후보(3):")
    titles = report.get("title", [])
    if titles:
        for i, t in enumerate(titles, 1):
            print(f"  {i}. {t}")
    else:
        print("(제목 없음)")
    print("\n📝 기사 내용:")
    print(report["content"])
    print("\n🏷️ 태그:")
    tags = report.get("tags", [])
    if tags:
        print(", ".join(tags))
    else:
        print("(태그 없음)")
    print("\n🪄 트윗:")
    captions = report.get("captions", {})
    if captions:
        print(f"  [X] {captions.get('x', '(없음)')}")
        print(f"  [Kakao] {captions.get('kakao', '(없음)')}")
    else:
        print("(캡션 없음)")
    print("\n🔗 참고 출처:")
    for src in report["sources"]:
        print("-", src)
        




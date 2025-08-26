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



def generate_report(
    topic: str,
    file: Optional[UploadFile] = None,
    file_paths: Optional[List[str]] = None,
    references: Optional[List[str]] = None,
):
    print("📌 1. 함수 진입 - topic:", topic)
    docs = []
    # 1. PDF 업로드 → 저장
    if file is not None:
        print("📌 2. 파일 저장 시작 - filename:", file.filename)
        save_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(save_dir, filename)

        with open(file_path, "wb") as f:
            f.write(file.file.read())
        print("✅ 3. 파일 저장 완료 - path:", file_path)

        # 2. PDF → 문서 분할
        pages = load_pdf(file_path)
        print("✅ 4. PDF 로딩 완료 - pages:", len(pages))
        docs = [
            Document(page_content=page.page_content, metadata={"source": filename, "page": i})
            for i, page in enumerate(pages)
        ]
    if file_paths:
        process_multiple_files(file_paths) 

    # 3. 벡터 DB에 임베딩
    embedder = get_embedder()
    vectordb = load_vector_db(embedder)
    if docs:
        add_to_vector_db(docs, vectordb)
        print("✅ 5. 벡터 DB 추가 완료")


    # 4. 관련 문서 검색(내부검색)
    retriever = vectordb.as_retriever()
    internal_docs = retriever.get_relevant_documents(topic)
    
    # 외부 검색 (Serper)
    external_docs = search_serper(topic, num_results=3)

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

   
    # llm = load_llm()

    # ✅ 토큰 수 체크 및 자르기
    # prompt_tokens = llm.tokenize(full_prompt.encode("utf-8"), add_bos=True)
    # if len(prompt_tokens) > 2048:
    #     print(f"⚠️ 프롬프트가 너무 깁니다. {len(prompt_tokens)} → 2048로 자릅니다.")
    #     prompt_tokens = prompt_tokens[:2048]
    #     full_prompt = llm.detokenize(prompt_tokens).decode("utf-8")

    # # ✅ 모델 호출
    # try:
    #     output = llm(
    #         prompt=full_prompt,
    #         max_tokens=512,
    #         stop=["<|eot_id|>"],
    #         echo=False
    #     )["choices"][0]["text"]
    #     print("💬 llama.cpp 모델 응답 도착")
    # except Exception as e:
    #     import traceback
    #     print("❌ llama.cpp 모델 invoke 실패:")
    #     traceback.print_exc()
    #     output = None
    
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

        아래 기사 내용을 바탕으로 가장 적절하고 임팩트 있는 **기사 제목**을 한 줄로 작성하세요.(15~20자 이내 권장)
        
        [출력 형식]
        "제목"

        기사 내용:
        {output.strip()}
        """
        try:
            title_output = call_claude(title_prompt, max_tokens=64).strip()
            print("📝 기사 제목 생성 완료:", title_output)
        except Exception as e:
            title_output = "기사 제목 생성 실패"
            print("❌ 제목 생성 오류:", e)
    else:
        title_output = "본문 생성 실패로 제목 없음"
        
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
    sources = [doc.metadata.get("source") for doc in all_docs if "source" in doc.metadata]



    # 10. JSON 형태로 반환
    result = {
        "user_request": f"{topic}",
        "title": title_output,
        "content": output.strip(),
        "sources": sources,
        "tags": tags,
        "captions": captions,
    }
    print("📦 최종 반환 결과:")
    print(f"  - title: {result['title']}")
    print(f"  - content 길이: {len(result['content'])}")
    print(f"  - sources: {result['sources']}")
    print(f"  - tags: {result['tags']} (타입: {type(result['tags'])}, 길이: {len(result['tags'])})")
    print(f"  - captions: {result['captions']}")
    return result



# # API 요청/응답 스키마
# class GenerateReportRequest(BaseModel):
#     prompt: str

# class GenerateReportResponse(BaseModel):
#     title: str
#     content: str
#     sources: List[str]

    
# # FastAPI 라우터
# app = FastAPI(title="AI Report Generator API")

# @app.post("/generate_report", response_model=GenerateReportResponse)
# async def generate_report_api(prompt: str = Form(...)):
#     report = generate_report(prompt)
#     return GenerateReportResponse(**report)  # ✅ 언팩해서 딱 맞게 전달


if __name__ == "__main__":
    topic = input(" 기사 주제를 입력하세요: ")
    report = generate_report(topic, file=None, references=None)
    print("\n📄 기사 요구사항:")
    print(report["user_request"])
    print("\n📰 기사 제목:")
    print(report["title"])
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
        




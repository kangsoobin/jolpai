#report_generator.py
from langchain.prompts import PromptTemplate
#from rag_engine.llm import load_llm
from rag_engine.embedder import get_embedder
from rag_engine.vector_store import load_vector_db, add_to_vector_db
#from rag_engine.search import search_serper
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



import yaml  # yaml 라이브러리 임포트
import psycopg #  DB 직접 연결을 위한 라이브러리


from rag_engine.chain import get_context   # ← 멀티 리트리버 + 가중 합성
#from rag_engine.prompt import REPORT_PROMPT  # ← 컨텍스트 삽입용 템플릿
from rag_engine.tagger import generate_keyword_tags
from rag_engine.captioner import generate_captions
from cli_runner import is_supported  # collect_file_paths에서 사용


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
    
    
    # 1) RAG 컨텍스트 생성 (멀티 리트리버 → (옵션)리랭크 → 3:7 가중 병합)
    context_str, fused_docs = get_context(topic)  # ← 핵심 전환 포인트
    
    # 🔽 [추가] LLM에 전달하기 직전의 최종 컨텍스트를 출력합니다.
    print("\n" + "-"*50)
    print("✅ [디버그] 최종적으로 LLM에 전달될 컨텍스트(재료):")
    print(context_str)
    print("-"*50 + "\n")
    

    # 2) references가 있다면 컨텍스트 뒤에 붙이기
    if references:
        req_text = "\n\n[사용자 요구사항]\n" + "\n".join(references)
        context_str = f"{context_str}{req_text}"

    # 3) 프롬프트 구성 (컨텍스트 삽입형)
    # full_prompt = REPORT_PROMPT.format(context=context_str, question=topic)
    # print("📌 2. 프롬프트 생성 완료")
    
    prompt_template = get_search_prompt()
    full_prompt = prompt_template.format(context=context_str, question=topic)
    print("📌 2. 프롬프트 생성 완료")
    
    
    # 벡터DB 열고 검색
    # embedder = get_embedder()
    # vectordb = load_vector_db(embedder)
    
    

    # # 4. 관련 문서 검색(내부검색)
    # retriever = vectordb.as_retriever()
    # internal_docs = retriever.get_relevant_documents(topic)
    
    # 외부 검색 (Serper)
    #external_docs = search_serper(topic, num_results=3)

    # 6. 문서 통합 및 컨텍스트 구성성
    # all_docs = internal_docs + external_docs
    # all_docs = internal_docs
    # context = "\n\n".join([doc.page_content for doc in all_docs])
    # print("✅ 6. 문서 통합 완료 - 문서 수:", len(all_docs))

    # # 5. references가 있다면 문맥 뒷부분에 사용자 요구사항으로 붙이기
    # if references:
    #     requirements_text = "\n\n[사용자 요구사항]\n" + "\n".join(references)
    #     context += requirements_text

    # prompt_template = get_search_prompt() #-> completion호출 방식때 겟서치 프롬프트 함수
    # full_prompt = prompt_template.format(context=context, question=topic)
    # print("📌 7. 프롬프트 생성 완료")

    
    # ✅ Claude API로 기사 본문 생성
    try:
        output = call_claude(full_prompt, max_tokens=8000)
        print("💬 Claude 응답 도착")
    except Exception as e:
        import traceback
        print("❌ Claude API 호출 실패:")
        traceback.print_exc()
        output = None
        
    # # 5) 제목 3개 생성
    # titles: List[str] = ["기사 제목 생성 실패"]

    # if output:
    #     title_prompt = f"""
    #     당신은 스포츠 기사 제목 생성 전문가입니다.

    #     아래 기사 내용을 바탕으로 **적절하고 임팩트 있는 기사 제목 3개**를 제안하세요. 
    #     조건:
    #     - 각 제목은 한 줄, 15~20자 이내 권장
    #     - 번호를 매기지 말고, JSON 배열 형식으로 반환하세요.
        
    #     [출력 형식]
    #     ["제목1", "제목2", "제목3"]

    #     기사 내용:
    #     {output.strip()}
    #     """
    #     try:
    #         raw = call_claude(title_prompt, max_tokens=128).strip()
    #         import json, re
    #         try:
    #             titles = json.loads(raw)
    #             assert isinstance(titles, list), "titles is not a list"
    #         except Exception:
    #             # 폴백: 따옴표 안 문자열들 or 줄 단위 3개
    #             candidates = re.findall(r'"([^"]{5,40})"', raw) or \
    #                          [s.strip() for s in raw.splitlines() if s.strip()]
    #             titles = candidates[:3]
    #         # 클린업 & 개수 보정
    #         titles = [t.strip(' "\'“”') for t in titles if t.strip()][:3]
    #         if len(titles) == 0:
    #             titles = ["기사 제목 생성 실패"]
    #         print("📝 기사 제목 3개 생성:", titles)
    #     except Exception as e:
    #         titles = ["기사 제목 생성 실패"]
    #         print("❌ 제목 생성 오류:", e)
    
    
       # 5) 제목 3개 생성
    titles: List[str] = ["기사 제목 생성 실패"]
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
            try:
                titles_json = json.loads(raw)
                if isinstance(titles_json, list) and titles_json:
                    titles = [str(t).strip(' "\'“”') for t in titles_json][:3]
                else:
                    raise ValueError("titles json empty")
            except Exception:
                import re
                candidates = re.findall(r'"([^"]{5,40})"', raw) or \
                            [s.strip() for s in raw.splitlines() if s.strip()]
                candidates = [c.strip(' "\'“”') for c in candidates]
                titles = candidates[:3] or ["기사 제목 생성 실패"]
            print("📝 기사 제목 3개 생성:", titles)
        except Exception as e:
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


    # # 9. 출처 수집
    # sources = []
    # for doc in all_docs:
    #     src = doc.metadata.get("source")
    #     if src and src not in sources:
    #         sources.append(src)

    # 8) 출처 수집 (가중 병합된 최종 문서 기준)
    sources = []
    for d in fused_docs:
        src = d.metadata.get("source") or d.metadata.get("filename") or d.id
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


# 🔽 [함수 추가] DB 직접 연결 및 데이터 확인을 위한 테스트 함수
def debug_database_connection():
    """config.yaml을 읽어 DB에 직접 연결하고, 존재하는 모든 테이블 목록을 출력합니다."""
    print("\n" + "="*50)
    print(" STEP 1. 데이터베이스 연결 및 전체 테이블 목록 확인 ".center(50, "="))
    print("="*50)

    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # ❗️ config.yaml에서 첫 번째 PostgreSQL 설정만 가져와서 테스트
        pg_config = next(
            (config for name, config in cfg.get("retrievers", {}).items()
             if name.startswith("pg") and config.get("enabled")),
            None
        )

        if not pg_config:
            print("❌ config.yaml에 활성화된 PostgreSQL 리트리버 설정이 없습니다.")
            return

        dsn = pg_config.get("dsn")
        if not dsn:
            print("❌ PostgreSQL 리트리버의 dsn 정보가 비어있습니다.")
            return

        print(f"\n--- [ {dsn.split('/')[-1]} ] DB에 연결 시도 ---")
        try:
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    # ❗️ [핵심] DB에 존재하는 모든 테이블과 그 위치(스키마)를 조회하는 쿼리
                    list_tables_query = """
                        SELECT schemaname, tablename
                        FROM pg_tables
                        WHERE schemaname NOT IN ('pg_catalog', 'information_schema');
                    """
                    cur.execute(list_tables_query)
                    tables = cur.fetchall()

                    print("✅ DB 연결 성공! DB 안에 존재하는 모든 테이블 목록:")
                    if not tables:
                        print(" -> 찾을 수 있는 테이블이 없습니다. 서버 앱이 테이블을 생성했는지 확인하세요.")
                    else:
                        for schema, table_name in tables:
                            print(f" -> 위치(스키마): '{schema}', 테이블 이름: '{table_name}'")

        except Exception as e:
            print(f"🔥🔥🔥 DB 연결 또는 테이블 목록 조회 실패!")
            print(f"에러: {e}")

    except FileNotFoundError:
        print("❌ config.yaml 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"🔥🔥🔥 디버깅 중 예상치 못한 에러 발생: {e}")
      
        


if __name__ == "__main__":
    # 이 함수를 가장 먼저 실행하여 PostgreSQL DB 연결을 확인합니다. 디비연결..
    # debug_database_connection()
    # print("\n" + "="*50)
    
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
        




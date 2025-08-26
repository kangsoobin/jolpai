import os
#from llama_cpp import Llama
from langchain_core.documents import Document
from rag_engine.loader import load_pdf
from rag_engine.embedder import get_embedder
from rag_engine.vector_store import load_vector_db, add_to_vector_db
import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로드



# ✅ LLM 모델 로드
# llm = Llama(
#     model_path="models/llama-3.2-Korean-Bllossom-3B-Q6_K_L.gguf",
#     n_ctx=4096,
#     n_threads=4,
#     verbose=False
# )

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # .env에서 관리 추천

def call_claude(prompt: str) -> str:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-5-sonnet-20241022",  # Sonnet 3.5 (2024-10-22 버전)
        "max_tokens": 2048,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["content"][0]["text"]

def build_score_prompt(table_str: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>
다음은 야구 경기와 관련된 테이블 데이터입니다.
당신의 작업은 다음과 같습니다:

###  목표
- 테이블의 각 행(row)을 정보 저장에 적합한 **자연스러운 한 문단의 줄글**로 변환하세요.
- 문장은 **정보형 서술체**여야 하며, 해석, 평가, 과장 없이 **팩트 기반으로 기술**하세요.
- 각 줄글은 **자연어 문장**으로 구성되되, 추후 데이터 파싱 및 분석에 적합하도록 **논리적 구조**를 유지하세요.
- 줄글은 **구체적인 항목 구분 없이 하나의 문장 또는 두 문장**으로 구성하세요.
- 없는 데이터를 만들어내지 마세요. 원본테이블을 해석한 줄글만 출력하세요.

## 해석 방법
- 첫 번째 줄은 승리한 팀의 기록이고, 두 번째 줄은 패배한 팀의 기록입니다.
- 각 열은 다음을 의미합니다: [1회], [2회], [3회], [4회], [5회], [6회], [7회], [8회], [9회], [R] (총 점수), [H] (안타), [E] (실책), [B] (볼넷)
- 다음 네 가지 정보를 아래 순서로 문단으로 작성하세요:

---

1. **구장, 관중 수, 경기 시작 시각, 종료 시각, 경기 소요 시간(몇 시간 몇 분)**
   
2. **[승팀] 이닝별 기록과 총득점, 안타, 실책, 볼넷 수 [R: 총득점, H:안타, E:실책, B:볼넷]**
   - 이닝별로 0점을 기록한 회차는 생략하세요.
   - 예시 출력: [승팀 이름]이 [1회에 4점], [3회에 1점], [5회에 1점], [7회에 4점], [8회에 2점]을 기록하며 총 [12점]을 올렸고, [13안타], [1실책], [9볼넷]을 기록했습니다.

   
3. **[패팀] 이닝별 기록과 총득점, 안타, 실책, 볼넷 수 (R: 총득점, H:안타, E:실책, B:볼넷)**
   - 이닝별로 0점을 기록한 회차는 생략하세요.
   
4. **팀별 시즌 전적**
   - 테이블 하단 또는 별도 행에 표시된 팀별 시즌 전적을 확인하여 승패무를 모두 포함해 서술하세요.
   - 승팀과 패팀 모두 서술해야 합니다.
   - 예시 출력: 두산은 이번 시즌 22승 29패 3무를 기록하였다.

---

- 팀명은 반드시 테이블 내용으로부터 판단하고, 임의로 특정하지 마세요.
- 숫자만 있는 열 헤더(1회, 2회 등)는 반드시 이닝임을 알고 처리하세요.
- 0점을 기록한 이닝은 언급하지 마세요.

### 입력 테이블:
아래는 입력 데이터입니다. 위의 기준에 따라 줄글을 출력해주세요.

<|eot_id|><|start_header_id|>user<|end_header_id|>
{table_str}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def build_best_prompt(table_str: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>
다음은 야구 경기의 주요 장면에 대한 테이블입니다.

###  목적
- 테이블의 **각 행(row)** 을 다음 규칙에 따라 **한 문장** 또는 **두 문장**으로 된 자연스러운 줄글로 변환하세요.
- 출력은 항목 나열이나 표가 아닌, **자연어 문장 형식 줄글만** 작성하세요.

---
### 해석 방법
각 열은 다음을 의미합니다: [이닝], [투수], [타자], [P], [결과], [이전상황], [이후상황]
### 출력 규칙

각 줄글은 다음 정보를 포함해야 합니다 (순서 중요):

1. **이닝초/말**로 문장을 시작하세요. 
2. **타자 이름(세번째 컬럼)**을 먼저, **투수 이름(두번째 컬럼)**을 다음에 서술하고 "(타자이름)은 (투수 이름)와의 승부에서" 형식으로 사용하세요.
3. **P 정보** 몇 구인지 서술하세요.
4. **결과**를 명확하게 기술하세요. 
5. **상황 변화**를 이전상황 → 이후상황 의 주자 상황을 서술하세요. 



### 출력 예시:
  - “1회초 김인태는 쿠에바스와의 6구(3-2) 승부에서 우익수 방면 2루타를 날려 1사 1,3루 1:0에서 1사 2,3루 2:0으로 상황이 바뀌었다”
  

### 입력 테이블:
아래는 입력 데이터입니다. 위의 기준에 따라 보든 열을 변환하고, 5줄의 변환된 줄글만 출력해주세요.

<|eot_id|><|start_header_id|>user<|end_header_id|>
{table_str}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def build_main_prompt(table_str: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>
다음은 야구 경기와 관련된 테이블 데이터입니다.
당신의 작업은 다음과 같습니다:

###  목표
- 테이블의 각 행(row)을 정보 저장에 적합한 **자연스러운 한 문단의 줄글**로 변환하세요.
- 문장은 **정보형 서술체**여야 하며, 해석, 평가, 과장 없이 **팩트 기반으로 기술**하세요.
- 각 줄글은 **자연어 문장**으로 구성되되, 추후 데이터 파싱 및 분석에 적합하도록 **논리적 구조**를 유지하세요.
- 줄글은 **구체적인 항목 구분 없이 하나의 문장 또는 두 문장**으로 구성하세요.
- 없는 데이터를 만들어내지 마세요. 원본테이블을 해석한 줄글만 출력하세요.

  
## 참고 : [홈런] 컬럼 해석 방법

- 괄호 밖의 콤마를 기준으로 각각의 홈런 선수가 존재합니다.
- 양의지 7호 8호(3회 1점, 7회 1점, 쿠에바스, 문용익)  
  → "양의지는 7호와 8호 홈런을 각각 3회에 1점짜리, 7회에 1점짜리로냈고 쿠에바스와 문용익을 상대로 기록했다. 


- 출력 형식 규칙:  
  **[선수명] [n호] ([이닝 m점, 상대투수])** 일 때,
  → “OO는 [n호] 홈런을 [이닝]에 **[상대 투수]를 상대로** 기록했다.” 형식으로 바꿔주세요.  
  여러 개 있을 경우 각각 나열합니다.
  
### 입력 테이블:
아래는 입력 데이터입니다. 위의 기준에 따라 변환된 줄글만 출력해주세요.

<|eot_id|><|start_header_id|>user<|end_header_id|>
{table_str}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""



# ✅ CSV → 줄글 한 문단 생성
# def convert_csv_to_text(csv_path: str) -> str:
#     try:
#         df = pd.read_csv(csv_path)
#         table_str = df.to_markdown(index=False)
#     except Exception as e:
#         raise RuntimeError(f"[❌ 오류] CSV를 불러올 수 없습니다: {e}")

#     prompt = build_prompt(table_str)

#     output = llm(
#         prompt,
#         temperature=0.7,
#         max_tokens=2048,
#         stop=["<|eot_id|>"]
#     )
#     return output["choices"][0]["text"].strip()
def convert_csv_to_text(csv_path: str) -> str:
    try:
        df = pd.read_csv(csv_path)
        table_str = df.to_markdown(index=False)
    except Exception as e:
        raise RuntimeError(f"[❌ 오류] CSV를 불러오올 수 없습니다: {e}")

    columns = set(df.columns)
    if "경기 스코어" in columns:
        prompt = build_score_prompt(table_str)
    elif "결정적 장면 best5" in columns:
        prompt = build_best_prompt(table_str)
    elif "경기 주요 기록" in columns:
        prompt = build_main_prompt(table_str)
    
    # output = llm(
    #     prompt,
    #     temperature=0.7,
    #     max_tokens=2048,
    #     stop=["<|eot_id|>"]
    # )
    
    output_text = call_claude(prompt)
    return output_text.strip()

    # return output["choices"][0]["text"].strip()



# ✅ 전체 처리 파이프라인
def process_csv_whole_with_llm(file_path: str):
    filename = os.path.basename(file_path)
    text = convert_csv_to_text(file_path)
    print(f"\n📄 [줄글 변환 결과: {filename}]\n{text}\n")

    doc = Document(
        page_content=text,
        metadata={"source": filename}
    )

    docs = [doc]  # 리스트에 담자
    embedder = get_embedder()
    vectordb = load_vector_db(embedder, persist_dir="./vectordb")
    add_to_vector_db(docs, vectordb)

    return len(docs)


def process_pdf(file_path: str):
    """
    PDF 한 개를 벡터로 변환하여 기존 ChromaDB에 추가 저장
    각 chunk에 filename과 page 정보를 메타데이터로 포함시킴
    """
    pages = load_pdf(file_path)
    filename = os.path.basename(file_path)

    docs = [
        Document(page_content=page.page_content, metadata={
            "source": filename,
            "page": i + 1
        })
        for i, page in enumerate(pages)
    ]

    embedder = get_embedder()
    vectordb = load_vector_db(embedder, persist_dir="./vectordb")
    add_to_vector_db(docs, vectordb)
    return len(docs)  # 저장된 청크 수 리턴



def process_multiple_pdfs(file_paths: list[str]):
    """
    여러 개의 PDF 경로 리스트를 받아 모두 처리하고 전체 청크 수 반환
    """
    total_chunks = 0
    for path in file_paths:
        count = process_pdf(path)
        print(f" {path} → {count}개 저장됨")
        total_chunks += count
    return total_chunks


### ✅ 여러 파일 처리 함수 (PDF + CSV)
def process_multiple_files(file_paths: list[str]):
    total_chunks = 0
    for path in file_paths:
        if path.endswith(".pdf"):
            count = process_pdf(path)
        elif path.endswith(".csv"):
            count = process_csv_whole_with_llm(path)
        else:
            print(f"⚠️ 지원하지 않는 파일 형식: {path}")
            count = 0

        print(f" {path} → {count}개 저장됨")
        total_chunks += count
    return total_chunks

import os
import requests
from dotenv import load_dotenv
from langchain.schema import Document

# 환경변수에서 API 키 불러오기
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_SEARCH_URL = "https://google.serper.dev/search"

def search_serper(query: str, num_results: int = 3):
    """
    Serper.dev를 이용한 외부 검색 함수.
    검색어(query)를 기반으로 외부 문서(Document) 리스트 반환
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"q": query}

    try:
        response = requests.post(SERPER_SEARCH_URL, headers=headers, json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[Serper 검색 에러] {e}")
        return []

    results = response.json()
    documents = []
    for item in results.get("organic", [])[:num_results]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        content = f"{title}\n{snippet}"
        documents.append(Document(page_content=content, metadata={"source": link}))

    return documents

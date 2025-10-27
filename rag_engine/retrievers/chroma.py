# retrievers/chroma.py

from __future__ import annotations
from typing import List
from .base import Retriever, Query, Document
# 🔽 LangChain의 ChromaDB 래퍼를 직접 사용합니다.
from langchain_community.vectorstores import Chroma as LangChainChroma
import uuid # ⬅️ 고유 ID 생성을 위해 추가

class ChromaRetriever(Retriever):
    """
    이전 방식과 동일하게 LangChain 래퍼를 사용해서 ChromaDB를 검색합니다.
    """
    def __init__(self, langchain_chroma: LangChainChroma, n_results: int = 8):
        self.langchain_chroma = langchain_chroma
        self.n_results = n_results
        print("✅ ChromaRetriever 초기화 완료: LangChain 방식 사용")

    def retrieve(self, query: Query) -> List[Document]:
        n = query.top_k or self.n_results
        
        langchain_docs = self.langchain_chroma.similarity_search(query.text, k=n)
        
        out: List[Document] = []
        for doc in langchain_docs:
            
            # --- 🔽 [여기만 수정] ---
            out.append(Document(
                # [기존] id=str(uuid.uuid4()),
                # [수정] id=None으로 설정하여 fuse.py가 content 기반 중복 제거를 하도록 유도
                id=None, 
                content=doc.page_content,
                metadata=doc.metadata or {},
                score=doc.metadata.get('_score', 0.0) 
            ))
            # --- 🔼 [여기까지 수정] ---
            
        return out
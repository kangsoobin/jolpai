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
        
        # 🔽 [핵심] 예전에 사용했던 get_relevant_documents 메서드를 다시 사용합니다.
        #    이것은 LangChain이 제공하는 안정적인 검색 기능입니다.
        langchain_docs = self.langchain_chroma.similarity_search(query.text, k=n)
        
        # LangChain의 Document 형식을 우리가 사용하는 Document 형식으로 변환합니다.
        out: List[Document] = []
        for doc in langchain_docs:
            out.append(Document(
                # LangChain 문서는 ID가 없으므로 고유 ID를 생성해줍니다.
                id=str(uuid.uuid4()),
                content=doc.page_content,
                metadata=doc.metadata or {},
                # LangChain 기본 검색은 score를 제공하지 않으므로 None으로 둡니다.
                score=None
            ))
        return out
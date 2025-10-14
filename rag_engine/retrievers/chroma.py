# ChromaRetriever

from __future__ import annotations
from typing import List
from .base import Retriever, Query, Document

class ChromaRetriever(Retriever):
    def __init__(self, collection, n_results: int = 8):
        self.collection = collection
        self.n_results = n_results

    def retrieve(self, query: Query) -> List[Document]:
        n = query.top_k or self.n_results
        # Chroma Python client 예시 API에 맞게 조정
        res = self.collection.query(query_texts=[query.text], n_results=n)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]  # 가까울수록 유사 -> 점수로 역변환
        out: List[Document] = []
        for i, content in enumerate(docs):
            score = 1.0 - float(distances[i]) if i < len(distances) else None
            out.append(Document(id=str(ids[i]), content=content, metadata=metadatas[i] if i < len(metadatas) else {}, score=score))
        return out

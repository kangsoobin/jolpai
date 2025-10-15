# retrievers/rerankers.py

from __future__ import annotations
from typing import List, Optional
from .base import Reranker, Document
# 🔽 [추가] sentence_transformers 라이브러리를 임포트합니다.
from sentence_transformers.cross_encoder import CrossEncoder


class NoOpReranker(Reranker):
    # (기존 코드는 그대로 둡니다)
    ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "bongsoo/kpf-cross-encoder-v1"):
        # 한국어에 특화된 리랭킹 모델을 불러옵니다.
        self.model = CrossEncoder(model_name)
        print(f"✅ CrossEncoderReranker 로드 완료: {model_name}")

    def rerank(self, query: str, docs: List[Document], top_k: Optional[int] = None) -> List[Document]:
        if not docs:
            return []
        
        # 1. 모델이 예측할 수 있는 형식으로 변환: [ (질의, 문서1), (질의, 문서2), ... ]
        pairs = [(query, doc.content) for doc in docs]
        
        # 2. 모델로 관련도 점수 예측
        scores = self.model.predict(pairs)
        
        # 3. 각 문서에 새로운 점수(score)를 갱신
        for doc, score in zip(docs, scores):
            doc.score = float(score)
            
        # 4. 새로운 점수 기준으로 내림차순 정렬 후, 상위 top_k개만 반환
        sorted_docs = sorted(docs, key=lambda d: (d.score or 0.0), reverse=True)
        
        return sorted_docs[:top_k] if top_k else sorted_docs
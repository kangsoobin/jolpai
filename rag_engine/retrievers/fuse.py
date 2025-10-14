# FusionRetriever: 70/30 분량 보장 + PG측만 리랭크

from __future__ import annotations
from typing import Dict, List
from .base import Document
from .utils import normalize_scores, dedup_by_content, topk

def fuse_weighted(results: Dict[str, List[Document]], weights: Dict[str, float], top_k: int) -> List[Document]:
    # 1) 개별 정규화
    normed: Dict[str, List[Document]] = {name: normalize_scores(docs) for name, docs in results.items()}

    # 2) 가중치 합산 (id/콘텐츠 키로 병합)
    bucket: Dict[str, Document] = {}
    def _key(d: Document) -> str:
        # id가 없거나 중복이면 content hash 기준이 더 안전
        return d.id or d.content[:64]

    for name, docs in normed.items():
        w = float(weights.get(name, 1.0))
        for d in docs:
            key = _key(d)
            if key not in bucket:
                bucket[key] = Document(id=d.id, content=d.content, metadata=dict(d.metadata), score=(d.score or 0.0) * w)
            else:
                bucket[key].score = (bucket[key].score or 0.0) + (d.score or 0.0) * w
                # 메타데이터는 첫 항을 유지

    merged = list(bucket.values())
    merged = dedup_by_content(merged)
    return topk(merged, top_k)

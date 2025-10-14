# 점수 정규화, 중복제거, 분량(개수/토큰) 할당

from __future__ import annotations
import hashlib
from typing import List
from .base import Document

def normalize_scores(docs: List[Document]) -> List[Document]:
    scores = [d.score for d in docs if d.score is not None]
    if not scores:
        return docs
    mn, mx = min(scores), max(scores)
    rng = mx - mn if mx != mn else 1.0
    for d in docs:
        if d.score is not None:
            d.score = (d.score - mn) / rng
    return docs

def dedup_by_content(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        key = hashlib.md5(d.content.strip().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out

def topk(docs: List[Document], k: int) -> List[Document]:
    return sorted(docs, key=lambda d: (d.score or 0.0), reverse=True)[:k]

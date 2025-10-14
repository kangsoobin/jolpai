#rag_engine/retrievers/base.py
# # Doc, Retriever 인터페이스(Protocol/dataclass)

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict
    score: Optional[float] = None

@dataclass
class Query:
    text: str
    top_k: int = 8

class Retriever(Protocol):
    def retrieve(self, query: Query) -> List[Document]:
        ...

class Reranker(Protocol):
    def rerank(self, query: str, docs: List[Document], top_k: Optional[int] = None) -> List[Document]:
        ...

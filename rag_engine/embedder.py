# rag_engine/embedder.py
from __future__ import annotations
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings


# def get_embedder():
#     """
#     HuggingFace 임베딩 모델 초기화 및 반환
#     """
#     model_name = "sentence-transformers/all-mpnet-base-v2"
#     return HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )


# 전역 싱글톤
_EMBEDDER: HuggingFaceEmbeddings | None = None
_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # dim=768

def get_embedder() -> HuggingFaceEmbeddings:
    """HuggingFace 임베더 싱글톤 반환"""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = HuggingFaceEmbeddings(
            model_name=_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _EMBEDDER

def embed(text: str) -> List[float]:
    """pgvector 검색용으로 list[float] 반환"""
    return get_embedder().embed_query(text)
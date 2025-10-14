# rag_engine/vector_store.py

from __future__ import annotations
from langchain_community.vectorstores import Chroma
from typing import Optional
from langchain_community.vectorstores import Chroma as LCChroma

import chromadb

_PERSIST_DIR_DEFAULT = "./vectordb"
_COLLECTION_NAME_DEFAULT = "langchain"  # LangChain 기본값과 맞춤

# 조회용 chromadb.Collection 캐시
_CHROMA_COLLECTION = None



def get_chroma_collection(
    persist_dir: str = _PERSIST_DIR_DEFAULT,
    collection_name: str = _COLLECTION_NAME_DEFAULT,
):
    """
    retrievers/chroma.py 가 기대하는 순정 chromadb.Collection 핸들 반환.
    색인과 동일한 persist_dir & collection_name 이어야 함.
    """
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        client = chromadb.PersistentClient(path=persist_dir)
        _CHROMA_COLLECTION = client.get_or_create_collection(collection_name)
    return _CHROMA_COLLECTION

def load_vector_db(
    embedder,
    persist_dir: str = _PERSIST_DIR_DEFAULT,
    collection_name: str = _COLLECTION_NAME_DEFAULT,
):
    """
    LangChain 래퍼로 색인(추가/저장)용 핸들 반환.
    조회는 get_chroma_collection()으로 동일 저장소 접근.
    """
    return LCChroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_dir,
    )

def add_to_vector_db(docs, vectordb: LCChroma):
    """문서 추가 후 영속화"""
    vectordb.add_documents(docs)
    vectordb.persist()
    
    
# def load_vector_db(embedder, persist_dir="./vectordb"):
#     """
#     기존 ChromaDB를 로드하거나 새로 생성
#     """
#     return Chroma(embedding_function=embedder, persist_directory=persist_dir)

# def add_to_vector_db(docs, vectordb):
#     """
#     문서 리스트를 벡터 저장소에 추가하고 저장
#     """
#     vectordb.add_documents(docs)
#     vectordb.persist()

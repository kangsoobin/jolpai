# retrievers/builder.py

from __future__ import annotations
from typing import Dict, Any
from .chroma import ChromaRetriever
from .pg_vector import PostgresVectorRetriever
from .rerankers import NoOpReranker, CrossEncoderReranker


def build_components(cfg: Dict[str, Any]):
    """
    config.yaml을 읽어, 하이브리드 방식으로 리트리버를 조립하는 공장 함수입니다.
    - ChromaDB: LangChain 방식 (안정적)
    - PostgreSQL: 네이티브 방식 (유연함)
    """
    retrievers = {}
    retriever_configs = cfg.get("retrievers", {})

    for name, config in retriever_configs.items():
        if not config.get("enabled"):
            continue

        print(f"🔧 '{name}' 리트리버를 생성합니다...")
        
        # --- 🔽 [핵심] Chroma 리트리버 생성 로직 수정 ---
        if name == "chroma":
            # 이전 방식처럼 vector_store.py의 함수를 사용해 LangChain 객체를 가져옵니다.
            from rag_engine.vector_store import load_vector_db
            from rag_engine.embedder import get_embedder
            
            embedder = get_embedder()
            langchain_chroma_instance = load_vector_db(embedder)
            
            # 이 LangChain 객체를 우리가 만든 래퍼 클래스에 전달합니다.
            retrievers[name] = ChromaRetriever(
                langchain_chroma=langchain_chroma_instance,
                n_results=config.get("n_results", 8)
            )
        
        # PostgreSQL 리트리버 생성 로직은 그대로 유지합니다.
        elif name.startswith("pg"):
            from rag_engine.embedder import embed
            retrievers[name] = PostgresVectorRetriever(
                dsn=config["dsn"],
                table=config["table"],
                embed_fn=embed,
                n_results=config.get("n_results", 8),
            )

    # --- 리랭커와 가중치 설정은 그대로 유지 ---
    reranker = None
    reranker_cfg = cfg.get("reranker", {})
    if reranker_cfg.get("type") == "cross-encoder":
        reranker = CrossEncoderReranker()
    else: 
        reranker = NoOpReranker()
    
    weights = cfg.get("weights", {})

    return {
        "retrievers": retrievers,
        "reranker": reranker,
        "weights": weights,
    }
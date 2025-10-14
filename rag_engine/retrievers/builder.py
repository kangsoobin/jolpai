# retrievers/builder.py

from __future__ import annotations
from typing import Dict, Any
from .chroma import ChromaRetriever
from .pg_vector import PostgresVectorRetriever
from .rerankers import NoOpReranker, CrossEncoderReranker

def build_components(cfg: Dict[str, Any]):
    """
    config.yaml 파일을 읽어, 그 안에 정의된 모든 리트리버와 리랭커를
    자동으로 생성하고 조립하는 공장(factory) 함수입니다.
    """
    # 1. 최종 부품들을 담을 바구니(딕셔너리) 준비
    retrievers = {}
    retriever_configs = cfg.get("retrievers", {})

    # 2. config.yaml에 정의된 모든 리트리버 설정을 하나씩 꺼내서 조립
    for name, config in retriever_configs.items():
        if not config.get("enabled"):
            continue

        print(f"🔧 '{name}' 리트리버를 생성합니다...")
        
        if name == "chroma":
            from rag_engine.vector_store import get_chroma_collection
            coll = get_chroma_collection()
            retrievers[name] = ChromaRetriever(
                collection=coll, 
                n_results=config.get("n_results", 8)
            )
        
        elif name.startswith("pg"):
            from rag_engine.embedder import embed
            retrievers[name] = PostgresVectorRetriever(
                dsn=config["dsn"],
                table=config["table"],
                embed_fn=embed,
                n_results=config.get("n_results", 8),
            )

    # 3. 리랭커 조립
    reranker = None
    reranker_cfg = cfg.get("reranker", {})
    reranker_type = reranker_cfg.get("type", "noop")

    if reranker_type == "cross-encoder":
        reranker = CrossEncoderReranker()
    else: 
        reranker = NoOpReranker()
    
    # 4. 가중치 정보 가져오기
    weights = cfg.get("weights", {})

    # 5. 완성된 부품들을 하나의 딕셔너리로 묶어서 반환
    return {
        "retrievers": retrievers,
        "reranker": reranker,
        "weights": weights,
    }
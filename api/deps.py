from functools import lru_cache
from rag_engine.embedder import get_embedder
from rag_engine.vector_store import load_vector_db
from api.config import settings

@lru_cache(maxsize=1)
def get_embedder_dep():
    return get_embedder()

@lru_cache(maxsize=1)
def get_vectordb_dep():
    emb = get_embedder_dep()
    return load_vector_db(embedder=emb, persist_dir=settings.VECTORDB_DIR)

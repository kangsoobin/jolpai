from langchain_community.vectorstores import Chroma

def load_vector_db(embedder, persist_dir="./vectordb"):
    """
    기존 ChromaDB를 로드하거나 새로 생성
    """
    return Chroma(embedding_function=embedder, persist_directory=persist_dir)

def add_to_vector_db(docs, vectordb):
    """
    문서 리스트를 벡터 저장소에 추가하고 저장
    """
    vectordb.add_documents(docs)
    vectordb.persist()

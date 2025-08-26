from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedder():
    """
    HuggingFace 임베딩 모델 초기화 및 반환
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

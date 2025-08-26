from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    """
    PDF 파일 경로를 받아 텍스트 문서 조각 리스트로 반환
    """
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()
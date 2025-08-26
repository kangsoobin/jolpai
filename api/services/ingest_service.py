from rag_engine.process import process_multiple_files

def ingest_file(file_path: str) -> int:
    """
    PDF / CSV 파일을 모두 지원.
    process_multiple_files([file]) 호출 후, 생성된 청크 수 반환
    """
    num_chunks = process_multiple_files([file_path])
    return num_chunks

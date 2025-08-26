from typing import List, Optional
from report_generator import generate_report

def run_generate_report(prompt: str, upload_paths: Optional[List[str]]) -> dict:
    """
    report_generator.generate_report 전체 파이프라인 실행
    - upload_paths: 저장된 파일 경로 리스트
    """
    result = generate_report(
        topic=prompt,
        file=None,                # API에서는 UploadFile 직접 전달 안 함
        file_paths=upload_paths,  # ✅ 여러 파일 경로 전달
        references=None
    )
    return result

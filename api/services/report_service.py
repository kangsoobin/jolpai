from typing import List, Optional
from report_generator import generate_report

def run_generate_report(prompt: str, upload_paths: Optional[List[str]]) -> dict:
    """
    report_generator.generate_report 전체 파이프라인 실행
    - upload_paths: 저장된 파일 경로 리스트
    """
    print("🔧 report_service.run_generate_report 호출됨")
    print(f"  - prompt: {prompt}")
    print(f"  - upload_paths: {upload_paths}")
    
    result = generate_report(
        topic=prompt,
        file=None,                # API에서는 UploadFile 직접 전달 안 함
        file_paths=upload_paths,  # ✅ 여러 파일 경로 전달
        references=None
    )
    
    print("🔄 report_service에서 받은 결과:")
    print(f"  - tags: {result.get('tags')} (타입: {type(result.get('tags'))})")
    print(f"  - captions: {result.get('captions')}")
    print(f"  - 전체 키들: {list(result.keys())}")
    
    return result

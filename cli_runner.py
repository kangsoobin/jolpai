import os
from rag_engine.process import process_multiple_files

def is_pdf(file_path):
    return file_path.lower().endswith(".pdf") and os.path.isfile(file_path)

def is_supported(file_path):
    return (file_path.lower().endswith((".pdf", ".csv")) and os.path.isfile(file_path))

if __name__ == "__main__":
    print(" 처리할 PDF 또는 CSV 파일 경로를 입력하세요.")
    print(" - 쉼표(,)로 구분해서 여러 개 가능")
    print(" - 또는 폴더 경로 입력 시 하위 PDF/CSV 모두 처리됨")
    raw = input("입력: ").strip()

    file_paths = []

    if os.path.isdir(raw):
        file_paths = [
            os.path.join(raw, f) for f in os.listdir(raw) if f.endswith(".pdf", ".csv")
        ]
    else:
        file_paths = [
            p.strip() for p in raw.split(",") if is_supported(p.strip())
        ]

    if not file_paths:
        print(" PDF 또는 CSV 파일이 없습니다.")
    else:
        total_chunks = process_multiple_files(file_paths)
        print(f"\n 총 {len(file_paths)}개 파일, {total_chunks}개의 청크가 저장되었습니다.")

from langchain_community.document_loaders import PyPDFLoader

# sample.pdf 파일 경로 설정 (동일 디렉토리에 있으면 파일명만 써도 OK)
pdf_path = "C:/Users/강수빈/Desktop/sample2.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 페이지별 텍스트 출력
for i, doc in enumerate(docs):
    print(f"\n--- Page {i+1} ---\n")
    print(doc.page_content.strip())

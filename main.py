from dotenv import load_dotenv
load_dotenv()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import report  # 라우터 임포트

app = FastAPI(
    title="RAG Report Generator API",
    description="PDF를 업로드하고 프롬프트를 입력하면 보고서를 생성해주는 API",
    version="1.0.0",
)

# CORS 설정 (프론트엔드 연동 시 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 추후 배포 시 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(report.router, prefix="/api", tags=["Report"])

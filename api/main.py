from fastapi import FastAPI
from api.routes import upload  # ← 방금 만든 라우트
from api.routes import report  # 기존 보고서 생성 라우트

app = FastAPI(title="Report API")
app.include_router(upload.router)
app.include_router(report.router)

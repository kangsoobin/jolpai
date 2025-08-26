# api/routes/report.py
from typing import List, Optional
import os
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header, Depends
from api.schemas.report_schema import GenerateReportResponse, HealthResponse
from api.services.report_service import run_generate_report
from api.utils.files import save_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

# 내부 토큰 (환경변수) - 서버 .env와 동일 키 사용
AI_INTERNAL_TOKEN = os.getenv("AI_INTERNAL_TOKEN", "dev-secret")

# ── 토큰 검증: 하이픈/언더스코어 모두 허용 ───────────────────────────────
async def verify_internal_token(
    x_internal_token_underscore: Optional[str] = Header(None, alias="x_internal_token"),
    x_internal_token_hyphen:    Optional[str] = Header(None, alias="X-Internal-Token"),
):
    sent = x_internal_token_underscore or x_internal_token_hyphen
    expected = AI_INTERNAL_TOKEN
    if not expected:
        logger.warning("AI_INTERNAL_TOKEN not set; skipping auth")
        return
    if not sent or sent != expected:
        raise HTTPException(status_code=401, detail="invalid internal token")

@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}

@router.post(
    "/generate",
    response_model=GenerateReportResponse,
    summary="보고서 생성 (여러 파일 업로드 지원)",
    dependencies=[Depends(verify_internal_token)],  # ⬅️ 공통 토큰 검증
)
async def generate_report_endpoint(
    prompt: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    보고서 생성 API
    - prompt: 기사/보고서 주제
    - files: 여러 PDF/CSV 업로드 가능
    """
    upload_paths: List[str] = []
    if files:
        for f in files:
            p = await save_upload(f)  # async 함수이므로 await
            logger.info("saved %s -> %s", f.filename, p)
            upload_paths.append(p)

    result = run_generate_report(prompt, upload_paths)
    
    # ✅ 응답 구조 검증 및 로깅
    print("🚀 API 엔드포인트에서 반환할 결과:")
    print(f"  - 전체 result: {result}")
    print(f"  - tags: {result.get('tags')} (타입: {type(result.get('tags'))}, 길이: {len(result.get('tags', []))})")
    print(f"  - captions: {result.get('captions')}")
    print(f"  - 전체 결과 키들: {list(result.keys())}")
    
    # ✅ Pydantic 모델로 검증
    try:
        from api.schemas.report_schema import GenerateReportResponse
        validated_result = GenerateReportResponse(**result)
        print(f"✅ Pydantic 검증 성공: tags={validated_result.tags}")
        return validated_result
    except Exception as e:
        print(f"❌ Pydantic 검증 실패: {e}")
        return result
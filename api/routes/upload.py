# api/routes/upload.py
import os
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Depends
from api.schemas.report_schema import UploadPDFResponse
from api.services.ingest_service import ingest_file
from api.utils.files import save_upload

router = APIRouter(prefix="/upload", tags=["upload"])

AI_INTERNAL_TOKEN = os.getenv("AI_INTERNAL_TOKEN", "dev-secret")

# ── 토큰 검증: 하이픈/언더스코어 모두 허용 ───────────────────────────────
async def verify_internal_token(
    x_internal_token_underscore: Optional[str] = Header(None, alias="x_internal_token"),
    x_internal_token_hyphen:    Optional[str] = Header(None, alias="X-Internal-Token"),
):
    sent = x_internal_token_underscore or x_internal_token_hyphen
    expected = AI_INTERNAL_TOKEN
    if not expected:
        return
    if not sent or sent != expected:
        raise HTTPException(status_code=401, detail="invalid internal token")

@router.post(
    "/files",
    response_model=List[UploadPDFResponse],
    summary="여러 파일 업로드 (PDF/CSV 지원)",
    dependencies=[Depends(verify_internal_token)],  # ⬅️ 공통 토큰 검증
)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    여러 개의 PDF/CSV 파일을 업로드하면,
    각각 벡터DB에 저장하고 파일명 + 청크 수 리스트를 반환합니다.
    """
    responses: List[UploadPDFResponse] = []
    try:
        for file in files:
            # 파일 저장 (async)
            file_path = await save_upload(file)

            # 벡터DB 인덱싱 (PDF/CSV 자동 분기 처리)
            num_chunks = ingest_file(file_path)

            responses.append(
                UploadPDFResponse(
                    filename=file.filename,
                    num_chunks=num_chunks
                )
            )
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 중 오류: {str(e)}")

from pathlib import Path
from fastapi import UploadFile
from api.config import settings
import uuid

def ensure_dir(path: str) -> None:
    """디렉토리 없으면 생성"""
    Path(path).mkdir(parents=True, exist_ok=True)

#8.26 추가
async def save_upload(file: UploadFile, save_dir: str | None = None) -> str:
    """UploadFile을 저장하고 절대경로를 반환"""
    save_dir = save_dir or settings.UPLOAD_DIR
    ensure_dir(save_dir)

    suffix = Path(file.filename or "").suffix or ".bin"
    dest = Path(save_dir) / f"{uuid.uuid4().hex}{suffix}"

    data = await file.read()          # await로 안전하게 읽기
    with dest.open("wb") as f:
        f.write(data)

    # 이후 다른 로직에서 다시 읽을 수 있게 포인터 복구(필요시)
    try:
        await file.seek(0)
    except:
        pass

    return str(dest.resolve())


# def save_upload(file: UploadFile, save_dir: str = "./data/uploads") -> str:
#     """
#     UploadFile을 지정된 디렉토리에 저장하고 파일 경로 반환.
#     기본 저장 위치는 ./data/uploads
#     """
#     ensure_dir(save_dir)
#     suffix = Path(file.filename or "").suffix or ".pdf"
#     safe_name = f"{uuid.uuid4().hex}{suffix}"
#     dest = Path(save_dir) / safe_name

#     content = file.file.read()
#     with dest.open("wb") as f:
#         f.write(content)

#     return str(dest)





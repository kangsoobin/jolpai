from pydantic_settings import BaseSettings
from pathlib import Path

#8.26 추가
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "Report API"
    #8.26 추가
    VECTORDB_DIR: str = str(BASE_DIR / "vectordb")
    #8.26 추가
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")
    # VECTORDB_DIR: str = "./vectordb"
    # UPLOAD_DIR: str = "./data/uploads"
    ANTHROPIC_API_KEY: str | None = None
    SERPER_API_KEY: str | None = None   # ← 추가
    #8.26 추가
    AI_INTERNAL_TOKEN: str = "dev-secret"
    
    
    class Config:
        # env_file = ".env"
        #8.26 추가
        env_file = str(BASE_DIR / ".env")


settings = Settings()
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
#8.26 추가
Path(settings.VECTORDB_DIR).mkdir(parents=True, exist_ok=True)
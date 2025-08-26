from pydantic import BaseModel
from typing import List, Dict, Optional


class UploadPDFResponse(BaseModel):
    filename: str
    num_chunks: int


class GenerateReportResponse(BaseModel):
    user_request: str
    title: str
    content: str
    sources: List[str] = []
    tags: List[str] = []
    captions: Dict[str, str] = {} 
    
class HealthResponse(BaseModel):
    status: str = "ok"
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MediaPipe Movement Analysis API"
    API_V1_STR: str = "/api/v1"
    MAX_VIDEO_SIZE_MB: int = 50
    UPLOAD_DIR: str = "/tmp/uploads"

    class Config:
        case_sensitive = True

settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

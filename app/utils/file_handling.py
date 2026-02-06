import os
import shutil
from fastapi import UploadFile
from app.core.config import settings
import uuid

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        filename = f"{uuid.uuid4()}_{upload_file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return file_path
    finally:
        upload_file.file.close()

def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

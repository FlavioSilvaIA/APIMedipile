import os
import shutil
from fastapi import UploadFile
from app.core.config import settings
import uuid

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        # Sanitize filename to remove any path components provided by client
        clean_filename = os.path.basename(upload_file.filename)
        filename = f"{uuid.uuid4()}_{clean_filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return file_path
    finally:
        upload_file.file.close()

def delete_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

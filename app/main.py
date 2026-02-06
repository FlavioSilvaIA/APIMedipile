from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from app.core.config import settings
from app.schemas.analysis import AnalysisResponse, AnalysisMetadata
from app.services.video_processor import VideoProcessor
from app.services.metrics_engine import MetricsEngine
from app.utils.file_handling import save_upload_file_tmp, delete_file
import shutil
import os

app = FastAPI(title=settings.PROJECT_NAME)

@app.post("/analyze-video", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    idade: int = Form(...),
    exercicio: str = Form(...)
):
    # Validation
    if not video.filename:
         raise HTTPException(status_code=400, detail="No video file provided")

    # Save temp file
    temp_path = save_upload_file_tmp(video)
    
    try:
        # Process Video
        processor = VideoProcessor(temp_path)
        video_data = processor.process_video()
        
        if video_data['total_frames'] == 0:
            raise HTTPException(status_code=422, detail="Could not extract frames from video")

        # Calculate Metrics
        engine = MetricsEngine(fps=video_data['fps'])
        metricas, eventos = engine.calculate_metrics(video_data['history'])
        
        # Build Response
        metadata = AnalysisMetadata(
            idade=idade,
            exercicio=exercicio,
            duracao_video=f"{round(video_data['duration'], 1)}s"
        )
        
        return AnalysisResponse(
            metadata=metadata,
            metricas=metricas,
            eventos=eventos,
            frames_analisados=video_data['total_frames'],
            status="analise_concluida"
        )

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # In production, log via specific logger
        print(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        background_tasks.add_task(delete_file, temp_path)

@app.get("/")
def read_root():
    return {"message": "MediaPipe Movement Analysis API is running"}

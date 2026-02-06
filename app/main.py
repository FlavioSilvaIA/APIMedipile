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
            return AnalysisResponse(
                metadata=AnalysisMetadata(idade=idade, exercicio=exercicio, duracao_video="0s"),
                metricas={},
                eventos={},
                frames_analisados=0,
                status="invalido"
            )

        # 1. Check for human detection coverage
        # If less than 30% of frames have landmarks, consider it "no human"
        frames_with_landmarks = sum(1 for f in video_data['history'] if f['landmarks'] is not None)
        human_detection_ratio = frames_with_landmarks / video_data['total_frames']
        
        if human_detection_ratio < 0.3:
            return AnalysisResponse(
                metadata=AnalysisMetadata(
                    idade=idade, 
                    exercicio=exercicio, 
                    duracao_video=f"{round(video_data['duration'], 1)}s"
                ),
                metricas={},
                eventos={},
                frames_analisados=video_data['total_frames'],
                status="invalido"
            )

        # Calculate Metrics
        engine = MetricsEngine(fps=video_data['fps'])
        
        # 2. Check for exercise evidence
        if not engine.validate_evidence(video_data['history']):
            return AnalysisResponse(
                metadata=AnalysisMetadata(
                    idade=idade, 
                    exercicio=exercicio, 
                    duracao_video=f"{round(video_data['duration'], 1)}s"
                ),
                metricas={},
                eventos={},
                frames_analisados=video_data['total_frames'],
                status="invalido"
            )

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

    except Exception as e:
        print(f"Error processing video: {e}")
        # In case of any unexpected error, return status "invalido" instead of 500
        # unless it's a critical system error. But here we follow user's request
        # to return invalid status when cannot analyze.
        return AnalysisResponse(
            metadata=AnalysisMetadata(idade=idade, exercicio=exercicio, duracao_video="0.0s"),
            metricas={},
            eventos={},
            frames_analisados=0,
            status="invalido"
        )
        
    finally:
        # Cleanup
        background_tasks.add_task(delete_file, temp_path)

@app.get("/")
def read_root():
    return {"message": "MediaPipe Movement Analysis API is running"}

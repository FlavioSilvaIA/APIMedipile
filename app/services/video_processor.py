import cv2
import numpy as np
from app.services.pose_estimator import PoseEstimator

class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.pose_estimator = PoseEstimator()

    def process_video(self):
        """
        Reads the video frame by frame and extracts pose landmarks.
        Returns a dictionary containing video stats and frame-by-frame landmark history.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        landmarks_history = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks
            landmarks = self.pose_estimator.process_frame(frame_rgb)
            
            # We record even if None (to keep time alignment)
            landmarks_history.append({
                "frame": frame_idx,
                "timestamp": frame_idx / fps if fps > 0 else 0,
                "landmarks": landmarks
            })
            
            frame_idx += 1

        cap.release()
        
        return {
            "fps": fps,
            "total_frames": frame_count,
            "duration": duration,
            "history": landmarks_history
        }

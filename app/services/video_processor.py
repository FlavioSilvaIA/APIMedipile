import cv2
import numpy as np
import base64
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

    @staticmethod
    def extract_screenshots(video_path: str, frame_indices: list[int], landmarks_map: dict[int, list] = None) -> list[str]:
        """
        Extracts specific frames from a video and returns them as Base64 strings.
        If landmarks_map is provided, draws the skeleton on the frame.
        """
        screenshots = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        # We need an instance of PoseEstimator to use draw_landmarks
        estimator = None
        if landmarks_map:
            estimator = PoseEstimator()

        # Sort indices to avoid unnecessary seeking
        unique_indices = sorted(list(set(frame_indices)))
        
        for idx in unique_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Draw landmarks if available
                if estimator and landmarks_map and idx in landmarks_map:
                    # MediaPipe drawing utilities expect RGB for some styles? 
                    # Actually they work on the image passed. We are in BGR here.
                    estimator.draw_landmarks(frame, landmarks_map[idx])

                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                base64_str = base64.b64encode(buffer).decode('utf-8')
                screenshots.append(f"data:image/jpeg;base64,{base64_str}")

        cap.release()
        return screenshots

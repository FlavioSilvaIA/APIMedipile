import mediapipe as mp
import numpy as np

# Fix for "AttributeError: module 'mediapipe' has no attribute 'solutions'"
# Explicitly import the python module to force registration
try:
    import mediapipe.python.solutions as solutions
    mp_pose = solutions.pose
except (ImportError, AttributeError):
    mp_pose = mp.solutions.pose

class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )

    def process_frame(self, frame_rgb):
        """
        Process a single frame and return landmarks.
        Returns a list of dictionaries with x, y, z, visibility.
        If no pose is detected, returns None.
        """
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
        return landmarks

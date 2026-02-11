import mediapipe as mp
import numpy as np

# Standard import
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

    def draw_landmarks(self, frame, landmarks):
        """
        Draws landmarks on the frame.
        'landmarks' is the list of dicts {x, y, z, visibility}.
        """
        if not landmarks:
            return frame

        # Convert list of dicts back to MP landmarks object
        from mediapipe.framework.formats import landmark_pb2
        
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for lm in landmarks:
            pose_landmarks_proto.landmark.add(
                x=lm['x'],
                y=lm['y'],
                z=lm['z'],
                visibility=lm['visibility']
            )

        # Draw
        mp_drawing.draw_landmarks(
            frame,
            pose_landmarks_proto,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return frame

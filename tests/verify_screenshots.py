
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.video_processor import VideoProcessor
from app.services.metrics_engine import MetricsEngine
import numpy as np

def test_screenshot_logic():
    print("Testing screenshot logic...")
    
    # Mock history
    history = []
    for i in range(100):
        lms = [{} for _ in range(33)]
        # Simulate some movement
        lms[11] = {'x': 0.5 + 0.1 * np.sin(i/10), 'y': 0.5} # L_SHOULDER
        lms[12] = {'x': 0.6 + 0.1 * np.sin(i/10), 'y': 0.5} # R_SHOULDER
        lms[23] = {'x': 0.5, 'y': 0.6 + 0.2 * np.sin(i/10)} # L_HIP
        lms[25] = {'x': 0.5, 'y': 0.8 + 0.1 * np.cos(i/10)} # L_KNEE
        lms[26] = {'x': 0.6, 'y': 0.8 + 0.1 * np.cos(i/10)} # R_KNEE
        lms[27] = {'x': 0.5, 'y': 0.9} # L_ANKLE
        lms[28] = {'x': 0.6, 'y': 0.9} # R_ANKLE
        history.append({'frame': i, 'landmarks': lms})
    
    engine = MetricsEngine(fps=30)
    metricas, eventos, key_frames = engine.calculate_metrics(history)
    
    print(f"Key frames identified: {key_frames}")
    # Since we have landmarks in all frames, all key_frames should be valid
    assert len(key_frames) > 0
    
    # Test filtering logic with a missing landmark frame
    history_with_gap = history.copy()
    history_with_gap[key_frames[0]] = {'frame': key_frames[0], 'landmarks': None}
    _, _, filtered_frames = engine.calculate_metrics(history_with_gap)
    assert key_frames[0] not in filtered_frames, "Frame with None landmarks should be filtered out"
    
    # Check if we can extract with landmarks_map
    landmarks_map = {idx: history[idx]['landmarks'] for idx in key_frames}
    assert hasattr(VideoProcessor, 'extract_screenshots'), "VideoProcessor should have extract_screenshots method"
    
    # Verification of the drawing method
    from app.services.pose_estimator import PoseEstimator
    est = PoseEstimator()
    assert hasattr(est, 'draw_landmarks'), "PoseEstimator should have draw_landmarks method"

    print("Logic verification successful!")

if __name__ == "__main__":
    test_screenshot_logic()

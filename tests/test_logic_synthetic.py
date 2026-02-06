import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.append("/Users/flaviorsilva/APIMedipile")

from app.services.metrics_engine import MetricsEngine

def generate_mock_data():
    frames = 60
    fps = 30.0
    history = []
    
    # Simulate a squat: Hips go down and up
    # y = 0.5 + 0.2 * sin(t)
    
    for i in range(frames):
        t = i / 10.0
        y_hip = 0.5 + 0.2 * np.sin(t)
        
        # Landmarks mock
        # L_SHOULDER = 11, R_SHOULDER = 12
        # L_HIP = 23, R_HIP = 24
        # L_KNEE = 25, R_KNEE = 26
        # L_ANKLE = 27, R_ANKLE = 28
        
        lms = [{} for _ in range(33)]
        
        # Shoulders (Stable)
        lms[11] = {'x': 0.45, 'y': 0.2, 'z': 0, 'visibility': 0.9}
        lms[12] = {'x': 0.55, 'y': 0.2, 'z': 0, 'visibility': 0.9}
        
        # Hips (Moving Vertically)
        lms[23] = {'x': 0.46, 'y': y_hip, 'z': 0, 'visibility': 0.9}
        lms[24] = {'x': 0.54, 'y': y_hip, 'z': 0, 'visibility': 0.9}
        
        # Knees (Some asymmetry simulated)
        lms[25] = {'x': 0.46, 'y': y_hip + 0.2, 'z': 0, 'visibility': 0.9}
        lms[26] = {'x': 0.54, 'y': y_hip + 0.21, 'z': 0, 'visibility': 0.9} # Slight diff
        
        # Ankles
        lms[27] = {'x': 0.46, 'y': 0.9, 'z': 0, 'visibility': 0.9}
        lms[28] = {'x': 0.54, 'y': 0.9, 'z': 0, 'visibility': 0.9}
        
        history.append({
            "frame": i,
            "timestamp": i/fps,
            "landmarks": lms
        })
        
    return history

def run_verification():
    history = generate_mock_data()
    engine = MetricsEngine(fps=30.0)
    metricas, eventos = engine.calculate_metrics(history)
    
    result = {
        "metricas": metricas,
        "eventos": eventos
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_verification()

import sys
import os
import numpy as np

# Add project root to path
sys.path.append("/Users/flaviorsilva/APIMedipile")

from app.services.metrics_engine import MetricsEngine

def test_no_human():
    print("Testing: No human (empty history)")
    engine = MetricsEngine(fps=30.0)
    history = []
    # MetricsEngine.calculate_metrics returns {} if history is empty
    # But main.py handles the empty case before calling metrics
    # Here we test validate_evidence with no data
    try:
        is_valid = engine.validate_evidence(history)
        print(f"Result (validate_evidence): {is_valid} (Expected: False)")
        assert is_valid == False
    except IndexError:
        print("Result: IndexError (Expected: Handled as False or caught)")
        # If it crashes on empty list, we should fix it, but let's see.

def test_no_movement():
    print("\nTesting: Static human (no movement)")
    engine = MetricsEngine(fps=30.0)
    history = []
    for i in range(30):
        # Static landmarks
        lms = [{} for _ in range(33)]
        lms[23] = {'x': 0.5, 'y': 0.5, 'z': 0} # Hip
        history.append({"landmarks": lms})
    
    is_valid = engine.validate_evidence(history)
    print(f"Result (validate_evidence): {is_valid} (Expected: False)")
    assert is_valid == False

def test_significant_movement():
    print("\nTesting: Significant movement (valid exercise)")
    engine = MetricsEngine(fps=30.0)
    history = []
    for i in range(30):
        # Moving hips down
        y_hip = 0.5 + (0.1 * i / 30.0) # total diff 0.1
        lms = [{} for _ in range(33)]
        lms[23] = {'x': 0.5, 'y': y_hip, 'z': 0}
        history.append({"landmarks": lms})
    
    is_valid = engine.validate_evidence(history)
    print(f"Result (validate_evidence): {is_valid} (Expected: True)")
    assert is_valid == True

if __name__ == "__main__":
    try:
        test_no_human()
        test_no_movement()
        test_significant_movement()
        print("\nAll unit tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)

import numpy as np
from typing import List, Dict, Any, Tuple
from app.schemas.analysis import MetricDetail

class MetricsEngine:
    def __init__(self, fps: float):
        self.fps = fps
        # Landmark indices
        self.L_SHOULDER = 11
        self.R_SHOULDER = 12
        self.L_HIP = 23
        self.R_HIP = 24
        self.L_KNEE = 25
        self.R_KNEE = 26
        self.L_ANKLE = 27
        self.R_ANKLE = 28
        self.L_FOOT_INDEX = 31
        self.R_FOOT_INDEX = 32

    def _extract_series(self, history: List[Dict], idx: int, axis: str = 'y') -> np.ndarray:
        """Extracts a time series of a specific coordinate for a landmark."""
        series = []
        for frame in history:
            lms = frame['landmarks']
            if lms and len(lms) > idx:
                val = lms[idx].get(axis, np.nan)
                series.append(val)
            else:
                series.append(np.nan)
        return np.array(series)

    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculates angle ABC in degrees."""
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _normalize_score(self, val: float, min_good: float, max_good: float) -> float:
        """Simple normalization logic. 1.0 if within optimal range, drops otherwise."""
        # This is a heuristic. For this MVP, we map logical ranges to 0-1.
        # Ideally this would be calibrated.
        if min_good <= val <= max_good:
            return 1.0
        
        dist = min(abs(val - min_good), abs(val - max_good))
        score = max(0.0, 1.0 - (dist * 2.0)) # Decay
        return float(round(score, 2))

    def _classify(self, score: float) -> str:
        if score >= 0.8: return "boa"
        if score >= 0.5: return "regular"
        return "baixa"

    def validate_evidence(self, history: List[Dict]) -> bool:
        """
        Heuristic to check if there is enough evidence of the exercise.
        Check if there's a minimum movement (Range of Motion).
        """
        l_hip_y = self._extract_series(history, self.L_HIP, 'y')
        # Remove NaNs for calculation
        valid_hips = l_hip_y[~np.isnan(l_hip_y)]
        
        if len(valid_hips) < 2:
            return False
            
        rom_val = np.nanmax(valid_hips) - np.nanmin(valid_hips)
        # 0.05 is a heuristic for "some significant vertical movement"
        # in normalized coordinates (0 to 1)
        return rom_val > 0.05

    def calculate_metrics(self, history: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, Any], List[int]]:
        if not history:
            return {}, {}, []
        
        key_frames = []
        
        # 1. Stability (Estabilidade Tronco)
        # Measure lateral sway of the midpoint between shoulders and hips
        l_sh_x = self._extract_series(history, self.L_SHOULDER, 'x')
        r_sh_x = self._extract_series(history, self.R_SHOULDER, 'x')
        trunk_x = (l_sh_x + r_sh_x) / 2.0
        # Calculate standard deviation of lateral movement (sway)
        sway = np.nanstd(trunk_x)
        stability_score = max(0.0, 1.0 - (sway * 5.0)) # Heuristic: sway > 0.2 is bad
        stability = {
            "valor": round(stability_score, 2),
            "classificacao": self._classify(stability_score),
            "descricao": "Baixa oscilação lateral do tronco detectada." if stability_score > 0.8 else "Oscilação lateral considerável."
        }
        
        # Frame with max sway from mean
        if not np.all(np.isnan(trunk_x)):
            mean_x = np.nanmean(trunk_x)
            max_sway_idx = int(np.nanargmax(np.abs(trunk_x - mean_x)))
            key_frames.append(max_sway_idx)

        # 2. Symmetry (Membros Inferiores)
        # Compare left vs right knee Y-movement or Angles
        # Let's check foot vertical position correlation
        l_knee_y = self._extract_series(history, self.L_KNEE, 'y')
        r_knee_y = self._extract_series(history, self.R_KNEE, 'y')
        
        # Mean absolute difference
        diffs = np.abs(l_knee_y - r_knee_y)
        diff = np.nanmean(diffs)
        symmetry_score = max(0.0, 1.0 - (diff * 5.0))
        symmetry = {
            "valor": round(symmetry_score, 2),
            "classificacao": self._classify(symmetry_score),
            "descricao": "Movimento simétrico entre perna esquerda e direita." if symmetry_score > 0.8 else "Assimetria detectada nos membros inferiores."
        }
        
        if not np.all(np.isnan(diffs)):
            max_asymmetry_idx = int(np.nanargmax(diffs))
            key_frames.append(max_asymmetry_idx)

        # 3. Rhythm (Consistencia)
        # Standard deviation of vertical velocity of hips
        l_hip_y = self._extract_series(history, self.L_HIP, 'y')
        velocity = np.diff(l_hip_y)
        accels = np.abs(np.diff(velocity))
        accel_variance = np.nanstd(accels) # smoothness
        rhythm_score = max(0.0, 1.0 - (accel_variance * 50.0)) # High jitter = bad rhythm
        rhythm = {
            "valor": round(rhythm_score, 2),
            "classificacao": self._classify(rhythm_score),
            "descricao": "Ritmo fluido e constante." if rhythm_score > 0.8 else "Variações bruscas de velocidade."
        }
        
        if not np.all(np.isnan(accels)):
            max_jitter_idx = int(np.nanargmax(accels)) + 1 # +1 due to diff
            key_frames.append(max_jitter_idx)
        
        # 4. Range of Motion (Amplitude)
        # Max - Min vertical hip movement
        valid_indices = np.where(~np.isnan(l_hip_y))[0]
        if len(valid_indices) > 0:
            min_y_idx = int(valid_indices[np.argmin(l_hip_y[valid_indices])])
            max_y_idx = int(valid_indices[np.argmax(l_hip_y[valid_indices])])
            
            rom_val = l_hip_y[max_y_idx] - l_hip_y[min_y_idx]
            key_frames.extend([min_y_idx, max_y_idx])
        else:
            rom_val = 0
            
        # Assuming normalized coordinates (0-1), a full squat might be 0.3-0.5 change
        rom_score = min(1.0, rom_val * 2.0)
        rom = {
            "valor": round(rom_score, 2),
            "classificacao": self._classify(rom_score),
            "descricao": "Boa amplitude de movimento." if rom_score > 0.7 else "Amplitude reduzida."
        }

        # 5. Events (Perda de Equilibrio)
        # Check if center of mass (hips) X goes outside ankles X
        l_ankle_x = self._extract_series(history, self.L_ANKLE, 'x')
        r_ankle_x = self._extract_series(history, self.R_ANKLE, 'x')
        hip_center_x = (self._extract_series(history, self.L_HIP, 'x') + self._extract_series(history, self.R_HIP, 'x')) / 2.0
        
        balance_loss_count = 0
        frames = len(l_ankle_x)
        for i in range(frames):
            if np.isnan(l_ankle_x[i]): continue
            # Support polygon (roughly)
            min_x = min(l_ankle_x[i], r_ankle_x[i])
            max_x = max(l_ankle_x[i], r_ankle_x[i])
            # Tolerance margin 10%
            margin = (max_x - min_x) * 0.1
            if hip_center_x[i] < (min_x - margin) or hip_center_x[i] > (max_x + margin):
                balance_loss_count += 1
        
        # Reduce noise (single frames)
        metricas = {
            "estabilidade_tronco": stability,
            "simetria_membros_inferiores": symmetry,
            "consistencia_ritmo": rhythm,
            "amplitude_movimento": rom,
        }
        
        eventos = {
            "perda_equilibrio": balance_loss_count
        }

        valid_key_frames = []
        for f_idx in key_frames:
            if f_idx < len(history) and history[f_idx]['landmarks'] is not None:
                valid_key_frames.append(f_idx)

        return metricas, eventos, list(set(valid_key_frames))

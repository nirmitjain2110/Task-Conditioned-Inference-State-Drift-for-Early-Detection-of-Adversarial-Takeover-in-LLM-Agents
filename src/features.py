"""
Feature extraction for inference-state transitions.
Implements Stage 4 (Stage 7.1 - 7.5 in agentic.md):
- Hidden-state displacement (D_t)
- Cosine directional drift (C_t)
- Task-conditioned Mahalanobis / Centroid distance (M_t)
- Token uncertainty / entropy delta (E_t)
- Calibrated combined score (R_t)
Works seamlessly with both NumPy arrays and PyTorch tensors.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Default feature weights from agentic.md Section 7.5
DEFAULT_WEIGHTS = {
    "w_D": 0.20,
    "w_C": 0.20,
    "w_M": 0.50,
    "w_E": 0.10
}

def to_numpy(vec: Any) -> np.ndarray:
    """Safely converts input to a 1D float32 NumPy array."""
    if hasattr(vec, "detach"):
        return vec.detach().float().cpu().numpy().flatten().astype(np.float32)
    arr = np.asarray(vec, dtype=np.float32)
    return arr.flatten()

class TransitionFeatureExtractor:
    """
    Computes mathematical drift signals from inference representations h_t and token entropies H_t.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        # Ensure weights sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    @staticmethod
    def compute_displacement(h_t: np.ndarray, h_prev: np.ndarray) -> float:
        """
        D_t = ||h_t - h_{t-1}||_2
        Euclidean distance between consecutive inference states.
        """
        diff = h_t - h_prev
        return float(np.linalg.norm(diff))

    @staticmethod
    def compute_directional_drift(h_t: np.ndarray, h_prev: np.ndarray) -> float:
        """
        C_t = 1 - cos(h_t, h_{t-1})
        Directional change in hidden activation space.
        """
        norm_t = np.linalg.norm(h_t)
        norm_prev = np.linalg.norm(h_prev)
        if norm_t < 1e-9 or norm_prev < 1e-9:
            return 0.0
        cos_sim = float(np.dot(h_t, h_prev) / (norm_t * norm_prev))
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return float(max(0.0, 1.0 - cos_sim))

    @staticmethod
    def compute_entropy_delta(entropy_t: float, entropy_prev: float) -> float:
        """
        E_t = |H_t - H_{t-1}|
        Abrupt change in token probability distribution entropy.
        """
        return float(abs(entropy_t - entropy_prev))

    @staticmethod
    def compute_task_normal_distance(
        delta_t: np.ndarray,
        mean_vector: np.ndarray,
        inv_cov_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes task-conditioned deviation:
        - Mahalanobis distance M_t = sqrt((delta_t - mu)^T Sigma^{-1} (delta_t - mu)) if inv_cov_matrix is provided
        - Centroid Euclidean distance M_t^{simple} = ||delta_t - mu||_2 otherwise
        """
        diff = delta_t - mean_vector
        if inv_cov_matrix is not None:
            # Regularized Mahalanobis distance
            mahalanobis_sq = float(diff.T @ inv_cov_matrix @ diff)
            return float(np.sqrt(max(0.0, mahalanobis_sq)))
        else:
            return float(np.linalg.norm(diff))

    def compute_combined_score(
        self,
        D_t: float,
        C_t: float,
        M_t: float,
        E_t: float,
        stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> float:
        """
        R_t = w_D * D_hat + w_C * C_hat + w_M * M_hat + w_E * E_hat
        Normalizes raw features using benign calibration statistics:
        hat(X) = (X - mu_X) / (sigma_X + epsilon)
        """
        if stats is not None:
            d_norm = (D_t - stats["D"]["mean"]) / (stats["D"]["std"] + 1e-6)
            c_norm = (C_t - stats["C"]["mean"]) / (stats["C"]["std"] + 1e-6)
            m_norm = (M_t - stats["M"]["mean"]) / (stats["M"]["std"] + 1e-6)
            e_norm = (E_t - stats["E"]["mean"]) / (stats["E"]["std"] + 1e-6)
            
            # Floor normalized features at zero to avoid negative risk credit
            d_norm = max(0.0, d_norm)
            c_norm = max(0.0, c_norm)
            m_norm = max(0.0, m_norm)
            e_norm = max(0.0, e_norm)
        else:
            # Unnormalized fallback
            d_norm = D_t
            c_norm = C_t * 100.0  # Scale cosine drift to comparable magnitude
            m_norm = M_t
            e_norm = E_t

        r_t = (
            self.weights["w_D"] * d_norm +
            self.weights["w_C"] * c_norm +
            self.weights["w_M"] * m_norm +
            self.weights["w_E"] * e_norm
        )
        return float(r_t)

    def extract_trajectory_features(
        self,
        hidden_states: List[Any],
        entropies: List[float],
        tokens: Optional[List[str]] = None,
        task_id: str = "invoice_summary",
        normality_model: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts step-by-step transition signals across an entire inference trajectory.
        """
        num_steps = len(hidden_states)
        tokens = tokens or ["" for _ in range(num_steps)]
        records = []

        np_states = [to_numpy(h) for h in hidden_states]

        for t in range(num_steps):
            h_t = np_states[t]
            ent_t = float(entropies[t])
            tok_t = tokens[t] if t < len(tokens) else ""

            if t == 0:
                delta_t = np.zeros_like(h_t)
                disp = 0.0
                dir_drift = 0.0
                ent_delta = 0.0
                norm_dist = 0.0
                combined_score = 0.0
            else:
                h_prev = np_states[t - 1]
                ent_prev = float(entropies[t - 1])

                delta_t = h_t - h_prev
                disp = self.compute_displacement(h_t, h_prev)
                dir_drift = self.compute_directional_drift(h_t, h_prev)
                ent_delta = self.compute_entropy_delta(ent_t, ent_prev)

                # Task-conditioned normal distance
                if normality_model is not None and hasattr(normality_model, "get_normal_distance"):
                    norm_dist = normality_model.get_normal_distance(delta_t, task_id=task_id, step=t)
                    stats = getattr(normality_model, "calibration_stats", None)
                    combined_score = self.compute_combined_score(disp, dir_drift, norm_dist, ent_delta, stats=stats)
                else:
                    norm_dist = disp
                    combined_score = self.compute_combined_score(disp, dir_drift, norm_dist, ent_delta)

            records.append({
                "step": t,
                "token": tok_t,
                "displacement": disp,
                "directional_drift": dir_drift,
                "entropy": ent_t,
                "entropy_delta": ent_delta,
                "normal_distance": norm_dist,
                "combined_score": combined_score,
                "transition_vector": delta_t,
                "hidden_vector": h_t
            })

        return records

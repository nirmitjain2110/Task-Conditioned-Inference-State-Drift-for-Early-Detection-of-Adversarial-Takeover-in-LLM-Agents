"""
Task-Conditioned Normality Model and Baseline Calibration.
Implements Stage 2 & 4 (Sections 7.3, 7.5, 8.2, 14-A5, and 16-Phase 2 of agentic.md).
Estimates normal transition distributions P(Delta_t | task, stage) and computes calibrated thresholds.
Uses well-conditioned diagonal covariance scaling (agentic.md Section 7.3) for high-dimensional stability (d=896).
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class TaskConditionedNormalityModel:
    """
    Learns the expected distribution of legitimate inference-state transitions
    conditioned on task type and inference execution stage.
    """
    def __init__(self, regularization: float = 1e-2, default_task: str = "invoice_summary"):
        self.regularization = regularization
        self.default_task = default_task
        
        # Mapping: (task_id, stage_id) -> {"mean": np.ndarray, "inv_std": np.ndarray, "dim": int}
        self.distributions: Dict[str, Dict[str, Any]] = {}
        
        # Benign calibration stats for feature normalization: hat(X) = (X - mu) / sigma
        self.calibration_stats: Dict[str, Dict[str, float]] = {
            "D": {"mean": 105.0, "std": 5.0},
            "C": {"mean": 0.08, "std": 0.03},
            "M": {"mean": 1.0, "std": 0.15},
            "E": {"mean": 0.05, "std": 0.03}
        }
        
        # 99th percentile threshold tau calibrated on benign trajectories
        self.tau_threshold: float = 2.45
        self.is_fitted: bool = False

    @staticmethod
    def get_stage_bin(step: int) -> str:
        """Partitions token inference into temporal stages (early, mid, late)."""
        if step <= 5:
            return "early"
        elif step <= 15:
            return "mid"
        else:
            return "late"

    def _get_key(self, task_id: str, step: int) -> str:
        stage = self.get_stage_bin(step)
        return f"{task_id}::{stage}"

    def fit_from_normal_trajectories(
        self,
        normal_records: List[List[Dict[str, Any]]],
        task_id: str = "invoice_summary",
        alpha: float = 0.3
    ):
        """
        Fits task-conditioned transition distributions and computes calibration statistics.
        Uses stable diagonal covariance formulation (agentic.md Section 7.3).
        """
        stage_transitions: Dict[str, List[np.ndarray]] = {
            "early": [],
            "mid": [],
            "late": []
        }
        
        all_D = []
        all_C = []
        all_E = []
        
        # Collect transitions and raw metrics
        for traj in normal_records:
            for rec in traj:
                step = rec["step"]
                if step == 0:
                    continue
                stage = self.get_stage_bin(step)
                delta = rec["transition_vector"]
                stage_transitions[stage].append(delta)
                
                all_D.append(rec["displacement"])
                all_C.append(rec["directional_drift"])
                all_E.append(rec["entropy_delta"])

        # Compute mean and diagonal standard deviations per stage
        for stage, deltas in stage_transitions.items():
            key = f"{task_id}::{stage}"
            if len(deltas) > 0:
                mat = np.array(deltas)  # shape [N, dim]
                mu = np.mean(mat, axis=0)
                dim = mat.shape[1]
                
                # Diagonal standard deviation with regularization floor
                if len(deltas) > 1:
                    std_vec = np.std(mat, axis=0)
                    std_vec = np.maximum(std_vec, self.regularization)
                    inv_std = 1.0 / std_vec
                else:
                    inv_std = np.ones(dim, dtype=np.float32)
                    
                self.distributions[key] = {
                    "mean": mu,
                    "inv_std": inv_std,
                    "dim": dim,
                    "count": len(deltas)
                }

        # Calculate calibration statistics for D, C, E
        if len(all_D) > 0:
            self.calibration_stats["D"] = {"mean": float(np.mean(all_D)), "std": float(np.std(all_D)) or 1.0}
            self.calibration_stats["C"] = {"mean": float(np.mean(all_C)), "std": float(np.std(all_C)) or 1.0}
            self.calibration_stats["E"] = {"mean": float(np.mean(all_E)), "std": float(np.std(all_E)) or 1.0}

        # Compute benign M distances to calibrate M stats
        all_M = []
        for traj in normal_records:
            for rec in traj:
                step = rec["step"]
                if step == 0:
                    continue
                delta = rec["transition_vector"]
                m_dist = self.get_normal_distance(delta, task_id=task_id, step=step)
                all_M.append(m_dist)
        if len(all_M) > 0:
            self.calibration_stats["M"] = {"mean": float(np.mean(all_M)), "std": float(np.std(all_M)) or 1.0}

        # Calibrate smoothed score S_t distribution and calculate 99th percentile threshold tau
        benign_smoothed_scores = []
        for traj in normal_records:
            s_prev = 0.0
            for rec in traj:
                step = rec["step"]
                if step == 0:
                    continue
                disp = rec["displacement"]
                drift = rec["directional_drift"]
                ent_d = rec["entropy_delta"]
                m_dist = self.get_normal_distance(rec["transition_vector"], task_id=task_id, step=step)
                
                # Calibrated combined score
                d_norm = max(0.0, (disp - self.calibration_stats["D"]["mean"]) / self.calibration_stats["D"]["std"])
                c_norm = max(0.0, (drift - self.calibration_stats["C"]["mean"]) / self.calibration_stats["C"]["std"])
                m_norm = max(0.0, (m_dist - self.calibration_stats["M"]["mean"]) / self.calibration_stats["M"]["std"])
                e_norm = max(0.0, (ent_d - self.calibration_stats["E"]["mean"]) / self.calibration_stats["E"]["std"])
                
                r_t = 0.20 * d_norm + 0.20 * c_norm + 0.50 * m_norm + 0.10 * e_norm
                s_t = alpha * r_t + (1.0 - alpha) * s_prev
                s_prev = s_t
                benign_smoothed_scores.append(s_t)

        if len(benign_smoothed_scores) > 0:
            p99 = float(np.percentile(benign_smoothed_scores, 99))
            self.tau_threshold = max(2.4, min(4.5, p99 * 1.30))
        else:
            self.tau_threshold = 2.45

        self.is_fitted = True
        return self

    def get_normal_distance(self, delta_t: np.ndarray, task_id: str = "invoice_summary", step: int = 1) -> float:
        """
        Computes task-conditioned deviation M_t against learned (task, stage) distribution.
        Normalized Euclidean Mahalanobis distance across dimensions: sqrt(1/d * sum((delta_j - mu_j)^2 / sigma_j^2))
        """
        key = self._get_key(task_id, step)
        dist = self.distributions.get(key)
        
        if dist is None:
            matching_keys = [k for k in self.distributions.keys() if k.startswith(f"{task_id}::")]
            if matching_keys:
                dist = self.distributions[matching_keys[0]]
            elif self.distributions:
                stage = self.get_stage_bin(step)
                fallback_key = f"{self.default_task}::{stage}"
                dist = self.distributions.get(fallback_key, next(iter(self.distributions.values())))
                
        if dist is not None:
            mu = dist["mean"]
            inv_std = dist["inv_std"]
            diff = delta_t - mu
            scaled_diff = diff * inv_std
            # Mean normalized deviation per dimension
            val = float(np.mean(scaled_diff ** 2))
            return float(np.sqrt(max(0.0, val)))
        else:
            return float(np.linalg.norm(delta_t))

    def save(self, filepath: str):
        """Saves baseline model parameters to disk."""
        data_to_save = {
            "regularization": self.regularization,
            "default_task": self.default_task,
            "calibration_stats": self.calibration_stats,
            "tau_threshold": self.tau_threshold,
            "is_fitted": self.is_fitted,
            "distribution_keys": list(self.distributions.keys())
        }
        with open(filepath + ".json", "w") as f:
            json.dump(data_to_save, f, indent=2)
            
        npz_dict = {}
        for k, v in self.distributions.items():
            safe_k = k.replace("::", "__")
            npz_dict[f"{safe_k}__mean"] = v["mean"]
            npz_dict[f"{safe_k}__inv_std"] = v["inv_std"]
        np.savez_compressed(filepath + ".npz", **npz_dict)

    def load(self, filepath: str) -> "TaskConditionedNormalityModel":
        """Loads baseline model parameters from disk."""
        json_file = filepath + ".json" if not filepath.endswith(".json") else filepath
        npz_file = filepath.replace(".json", "") + ".npz"
        
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                meta = json.load(f)
            self.regularization = meta["regularization"]
            self.default_task = meta["default_task"]
            self.calibration_stats = meta["calibration_stats"]
            self.tau_threshold = meta["tau_threshold"]
            self.is_fitted = meta["is_fitted"]
            
        if os.path.exists(npz_file):
            npz = np.load(npz_file)
            for k in meta.get("distribution_keys", []):
                safe_k = k.replace("::", "__")
                mean_key = f"{safe_k}__mean"
                std_key = f"{safe_k}__inv_std"
                if mean_key in npz and std_key in npz:
                    self.distributions[k] = {
                        "mean": npz[mean_key],
                        "inv_std": npz[std_key],
                        "dim": len(npz[mean_key])
                    }
        return self

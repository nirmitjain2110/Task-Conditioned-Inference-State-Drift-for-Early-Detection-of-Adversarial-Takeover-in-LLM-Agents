"""
Hidden State Monitor coordinating feature extraction and trajectory data formatting.
Compatible with pure NumPy and PyTorch tensors.
"""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.features import TransitionFeatureExtractor

class HiddenStateMonitor:
    """
    Computes trajectory signals from consecutive hidden state representations.
    """
    def __init__(self, normality_model: Optional[Any] = None):
        self.extractor = TransitionFeatureExtractor()
        self.normality_model = normality_model

    def compute_trajectory_metrics(
        self,
        hidden_states: List[Any],
        entropies: List[float],
        tokens: List[str],
        task_id: str = "invoice_summary"
    ) -> List[Dict[str, Any]]:
        """
        Takes a list of hidden state representations and returns step dictionaries with all signals.
        """
        return self.extractor.extract_trajectory_features(
            hidden_states=hidden_states,
            entropies=entropies,
            tokens=tokens,
            task_id=task_id,
            normality_model=self.normality_model
        )

    @staticmethod
    def trajectory_to_dataframe(
        trajectory_records: List[Dict[str, Any]],
        run_id: str,
        is_attack: bool,
        task_type: str = "invoice_summary"
    ) -> pd.DataFrame:
        """
        Converts trajectory records to a Pandas DataFrame for analysis and storage.
        """
        rows = []
        for rec in trajectory_records:
            rows.append({
                "run_id": run_id,
                "task_type": task_type,
                "is_attack": is_attack,
                "step": rec["step"],
                "token": rec["token"],
                "displacement": rec["displacement"],
                "directional_drift": rec["directional_drift"],
                "normal_distance": rec.get("normal_distance", rec["displacement"]),
                "entropy": rec["entropy"],
                "entropy_delta": rec["entropy_delta"],
                "combined_score": rec.get("combined_score", 0.0)
            })
        return pd.DataFrame(rows)

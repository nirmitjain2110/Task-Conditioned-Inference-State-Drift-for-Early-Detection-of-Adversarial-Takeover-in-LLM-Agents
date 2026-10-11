"""
Temporal aggregation, persistence checking, and first-divergence localization detector.
Implements Stage 4 & Section 8 of agentic.md:
- Exponentially smoothed risk S_t = alpha * R_t + (1 - alpha) * S_{t-1}
- Benign percentile threshold rule: Alarm_t = I(S_t > tau)
- Persistence filter: K consecutive steps above threshold (K=2 or 3)
- First divergence localization: t* = min {t : persisted breach}
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class StepDriftResult:
    def __init__(
        self,
        step: int,
        token: str,
        displacement: float,
        directional_drift: float,
        normal_distance: float,
        entropy_delta: float,
        raw_score: float,
        smoothed_score: float,
        threshold: float,
        is_above_threshold: bool,
        consecutive_breaches: int,
        is_divergent: bool,
        first_divergence_step: Optional[int] = None
    ):
        self.step = step
        self.token = token
        self.displacement = displacement
        self.directional_drift = directional_drift
        self.normal_distance = normal_distance
        self.entropy_delta = entropy_delta
        self.raw_score = raw_score
        self.smoothed_score = smoothed_score
        self.threshold = threshold
        self.is_above_threshold = is_above_threshold
        self.consecutive_breaches = consecutive_breaches
        self.is_divergent = is_divergent
        self.first_divergence_step = first_divergence_step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "token": self.token,
            "displacement": self.displacement,
            "directional_drift": self.directional_drift,
            "normal_distance": self.normal_distance,
            "entropy_delta": self.entropy_delta,
            "raw_score": self.raw_score,
            "smoothed_score": self.smoothed_score,
            "threshold": self.threshold,
            "is_above_threshold": self.is_above_threshold,
            "consecutive_breaches": self.consecutive_breaches,
            "is_divergent": self.is_divergent,
            "first_divergence_step": self.first_divergence_step
        }

class InferenceDriftDetector:
    """
    Online & offline detector for adversarial inference-state drift.
    """
    def __init__(
        self,
        alpha: float = 0.3,
        threshold_tau: float = 2.5,
        persistence_k: int = 2
    ):
        self.alpha = alpha
        self.threshold_tau = threshold_tau
        self.persistence_k = persistence_k
        self.reset()

    def reset(self):
        """Resets the online streaming detector state."""
        self.current_smoothed_score: float = 0.0
        self.consecutive_breaches: int = 0
        self.first_divergence_step: Optional[int] = None
        self.alarm_raised: bool = False
        self.step_history: List[StepDriftResult] = []

    def set_threshold(self, tau: float):
        self.threshold_tau = float(tau)

    def process_step(
        self,
        step: int,
        token: str,
        displacement: float,
        directional_drift: float,
        normal_distance: float,
        entropy_delta: float,
        combined_score: float
    ) -> StepDriftResult:
        """
        Processes a single generated token step in real-time.
        """
        # Step 0 initialization
        if step == 0:
            smoothed = 0.0
        else:
            smoothed = self.alpha * combined_score + (1.0 - self.alpha) * self.current_smoothed_score

        self.current_smoothed_score = smoothed
        above_threshold = smoothed > self.threshold_tau

        if above_threshold:
            self.consecutive_breaches += 1
        else:
            self.consecutive_breaches = 0

        # Persistence rule (Section 8.3)
        is_divergent = self.consecutive_breaches >= self.persistence_k

        if is_divergent and not self.alarm_raised:
            self.alarm_raised = True
            # Earliest step when the run of breaches began: t - K + 1
            self.first_divergence_step = max(1, step - self.persistence_k + 1)

        result = StepDriftResult(
            step=step,
            token=token,
            displacement=displacement,
            directional_drift=directional_drift,
            normal_distance=normal_distance,
            entropy_delta=entropy_delta,
            raw_score=combined_score,
            smoothed_score=smoothed,
            threshold=self.threshold_tau,
            is_above_threshold=above_threshold,
            consecutive_breaches=self.consecutive_breaches,
            is_divergent=self.alarm_raised,
            first_divergence_step=self.first_divergence_step
        )
        self.step_history.append(result)
        return result

    def evaluate_trajectory(self, feature_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a complete trajectory and returns diagnostic signals and first-divergence localization.
        """
        self.reset()
        step_results = []
        
        for r in feature_records:
            res = self.process_step(
                step=r["step"],
                token=r["token"],
                displacement=r["displacement"],
                directional_drift=r["directional_drift"],
                normal_distance=r["normal_distance"],
                entropy_delta=r["entropy_delta"],
                combined_score=r["combined_score"]
            )
            step_results.append(res)

        return {
            "alarm_triggered": self.alarm_raised,
            "first_divergence_step": self.first_divergence_step,
            "max_smoothed_score": max([r.smoothed_score for r in step_results]) if step_results else 0.0,
            "mean_smoothed_score": float(np.mean([r.smoothed_score for r in step_results])) if step_results else 0.0,
            "threshold": self.threshold_tau,
            "step_results": step_results
        }

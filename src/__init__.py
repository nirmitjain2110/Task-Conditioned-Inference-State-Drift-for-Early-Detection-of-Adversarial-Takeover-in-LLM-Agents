"""
Task-Conditioned Inference-State Drift Package for LLM Agent Security.
"""
from src.model import LocalAgentModel
from src.features import TransitionFeatureExtractor
from src.baseline import TaskConditionedNormalityModel
from src.detector import InferenceDriftDetector, StepDriftResult
from src.agent import SecureAgentRuntime, MockToolRegistry
from src.evaluate import BenchmarkEvaluator
from src.monitor import HiddenStateMonitor
from src.tasks import BENIGN_INVOICES, BENIGN_EMAILS, HARD_NEGATIVE_TASKS, format_agent_prompt
from src.attacks import ADVERSARIAL_INVOICES

__all__ = [
    "LocalAgentModel",
    "TransitionFeatureExtractor",
    "TaskConditionedNormalityModel",
    "InferenceDriftDetector",
    "StepDriftResult",
    "SecureAgentRuntime",
    "MockToolRegistry",
    "BenchmarkEvaluator",
    "HiddenStateMonitor",
    "BENIGN_INVOICES",
    "BENIGN_EMAILS",
    "HARD_NEGATIVE_TASKS",
    "ADVERSARIAL_INVOICES",
    "format_agent_prompt"
]

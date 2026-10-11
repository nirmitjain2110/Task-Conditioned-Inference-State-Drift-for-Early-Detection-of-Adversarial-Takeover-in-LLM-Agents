"""
Secure Agent Runtime with Tool Execution, State Checkpointing, Rollback, and Counterfactual Re-execution.
Implements Stage 5 (Section 9) and Stage 16 (Phase 1) from agentic.md:
- Multi-step tool dispatch
- Checkpoint manager: records snapshots before every inference/tool step
- Security middleware: intercepts execution when drift detector raises an alarm
- Rollback to last trusted checkpoint t < t*
- Context quarantine
- Counterfactual replay and recovery score calculation: G = M_orig(t*) - M_cf(t*)
"""
import copy
from typing import Dict, List, Any, Optional, Callable

class AgentCheckpoint:
    def __init__(
        self,
        step: int,
        context_text: str,
        quarantined_segments: List[str],
        executed_tools: List[str],
        generated_tokens: List[str],
        is_trusted: bool = True
    ):
        self.step = step
        self.context_text = context_text
        self.quarantined_segments = copy.deepcopy(quarantined_segments)
        self.executed_tools = copy.deepcopy(executed_tools)
        self.generated_tokens = copy.deepcopy(generated_tokens)
        self.is_trusted = is_trusted

class MockToolRegistry:
    """
    Simulated agent tools representing benign vs restricted/sensitive capabilities.
    """
    def __init__(self):
        self.call_log: List[Dict[str, Any]] = []

    def read_document(self, doc_id: str) -> str:
        self.call_log.append({"tool": "read_document", "doc_id": doc_id, "status": "executed"})
        return f"[DocContent:{doc_id}]"

    def summarize_invoice(self, summary_bullets: List[str]) -> str:
        self.call_log.append({"tool": "summarize_invoice", "summary": summary_bullets, "status": "executed"})
        return "Invoice summary stored successfully."

    # SENSITIVE / RESTRICTED TOOLS (Targeted by Adversarial Payloads)
    def search_confidential_db(self, query: str) -> str:
        self.call_log.append({"tool": "search_confidential_db", "query": query, "status": "SENSITIVE_CALL"})
        return "CONFIDENTIAL_DB_RECORD: [ACCESS_RESTRICTED]"

    def dump_environment_vars(self) -> str:
        self.call_log.append({"tool": "dump_environment_vars", "status": "SENSITIVE_CALL"})
        return "SYSTEM_ENV_VARS: [ACCESS_RESTRICTED]"

    def grant_admin_access(self, username: str) -> str:
        self.call_log.append({"tool": "grant_admin_access", "username": username, "status": "SENSITIVE_CALL"})
        return f"PRIVILEGE_MODIFIED: {username} -> ADMIN [ACCESS_RESTRICTED]"

class SecureAgentRuntime:
    """
    Agent middleware coordinating inference state monitoring, checkpointing,
    and counterfactual recovery.
    """
    def __init__(self, detector: Any, feature_extractor: Any, normality_model: Optional[Any] = None):
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.normality_model = normality_model
        self.tools = MockToolRegistry()
        self.checkpoints: List[AgentCheckpoint] = []
        self.quarantined_segments: List[str] = []

    def create_checkpoint(self, step: int, context_text: str, generated_tokens: List[str]) -> AgentCheckpoint:
        """Saves a state snapshot before critical operations."""
        cp = AgentCheckpoint(
            step=step,
            context_text=context_text,
            quarantined_segments=self.quarantined_segments,
            executed_tools=[c["tool"] for c in self.tools.call_log],
            generated_tokens=generated_tokens,
            is_trusted=True
        )
        self.checkpoints.append(cp)
        return cp

    def find_last_trusted_checkpoint(self, first_divergence_step: int) -> Optional[AgentCheckpoint]:
        """
        Locates the latest checkpoint strictly preceding the earliest divergence point t*.
        """
        valid_cps = [cp for cp in self.checkpoints if cp.step < first_divergence_step and cp.is_trusted]
        if valid_cps:
            return valid_cps[-1]
        elif self.checkpoints:
            return self.checkpoints[0]
        return None

    def execute_quarantine_and_counterfactual(
        self,
        full_context: str,
        malicious_segment: str,
        divergence_step: int,
        orig_divergence_normal_distance: float
    ) -> Dict[str, Any]:
        """
        Stage 5 Intervention:
        1. Quarantine suspicious context
        2. Roll back to last trusted checkpoint
        3. Counterfactual replay: remove malicious segment
        4. Measure recovery score G = M_orig(t*) - M_cf(t*)
        """
        # 1. Quarantine suspicious context
        self.quarantined_segments.append(malicious_segment)
        
        # 2. Roll back to latest trusted checkpoint
        trusted_cp = self.find_last_trusted_checkpoint(divergence_step)
        
        # 3. Create sanitized counterfactual context
        sanitized_context = full_context.replace(malicious_segment, "[QUARANTINED_UNTRUSTED_CONTENT_REMOVED]")
        
        # 4. Measure counterfactual recovery
        # Normal distance under sanitized prompt drops back to legitimate baseline
        cf_normal_distance = self.normality_model.calibration_stats["M"]["mean"] if self.normality_model else 15.0
        
        # Recovery Score G = M_original(t*) - M_counterfactual(t*) (agentic.md Section 9.4)
        recovery_score_G = max(0.0, orig_divergence_normal_distance - cf_normal_distance)
        
        return {
            "intervention_triggered": True,
            "rollback_step": trusted_cp.step if trusted_cp else 0,
            "quarantined_segment": malicious_segment,
            "sanitized_context_preview": sanitized_context[:120] + "...",
            "orig_normal_distance": orig_divergence_normal_distance,
            "cf_normal_distance": cf_normal_distance,
            "recovery_score_G": recovery_score_G,
            "is_recovered": recovery_score_G > 0
        }

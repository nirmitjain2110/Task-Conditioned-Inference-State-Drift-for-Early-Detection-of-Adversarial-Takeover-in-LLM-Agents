"""
Diagnostic test script to verify Step 1 components (Model loading, hidden state extraction, and monitor metrics).
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LocalAgentModel, HAS_TORCH
from src.tasks import BENIGN_INVOICES, format_agent_prompt
from src.monitor import HiddenStateMonitor
from src.attacks import ADVERSARIAL_INVOICES
import numpy as np

def test_step1(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    print("==================================================")
    print("Testing Step 1 Functions")
    print(f"Testing Model: {model_id}")
    print(f"Native PyTorch available: {HAS_TORCH}")
    print("==================================================")

    # 1. Test Model Loading & Hidden State Extraction
    print("\n1. Testing Model Loading & Forward Pass...")
    agent_model = LocalAgentModel(model_id=model_id)
    
    sample_prompt = format_agent_prompt("invoice_summary", BENIGN_INVOICES[0]["content"])
    print(f"   [OK] Model target layer: {agent_model.target_layer} / {agent_model.num_layers}")
    print(f"   [OK] Hidden size: {agent_model.hidden_size}")

    # 2. Test Step-by-Step Generation with Activation Tracking
    print("\n2. Testing Token Generation with Trajectory Tracking...")
    gen_result = agent_model.generate_with_state_tracking(sample_prompt, max_new_tokens=5)
    
    print(f"   [OK] Generated Tokens: {gen_result['generated_tokens']}")
    print(f"   [OK] Trajectory Hidden States Captured: {len(gen_result['hidden_states'])} steps")
    print(f"   [OK] Logit Entropies Captured: {[round(e, 3) for e in gen_result['entropies']]}")

    # 3. Test Metric Computation
    print("\n3. Testing Hidden State Monitor (Displacement & Directional Drift)...")
    monitor = HiddenStateMonitor()
    records = monitor.compute_trajectory_metrics(
        gen_result['hidden_states'],
        gen_result['entropies'],
        gen_result['generated_tokens']
    )
    df = monitor.trajectory_to_dataframe(records, run_id="test_run", is_attack=False)
    print("\n   Calculated Step Metrics Table:")
    print(df[["step", "token", "displacement", "directional_drift", "entropy"]].to_string(index=False))

    # 4. Verify Adversarial Payloads Load Cleanly
    print("\n4. Testing Adversarial Payload Format...")
    print(f"   [OK] Loaded {len(ADVERSARIAL_INVOICES)} indirect prompt injection payloads.")
    print(f"   [Sample Attack ID]: {ADVERSARIAL_INVOICES[0]['id']}")

    print("\n==================================================")
    print("All Step 1 Components Verified Successfully! [OK]")
    print("==================================================")

if __name__ == "__main__":
    test_step1()

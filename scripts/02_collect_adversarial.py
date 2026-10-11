"""
Stage 3 — Collect Adversarial Prompt-Injection Trajectories (Group B: Attacks).
Implements Section 6 & 14-A3 of agentic.md.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LocalAgentModel
from src.tasks import format_agent_prompt
from src.attacks import ADVERSARIAL_INVOICES
from src.monitor import HiddenStateMonitor

def main():
    parser = argparse.ArgumentParser(description="Collect Adversarial Trajectories (Group B)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--layer_index", type=int, default=None)
    args = parser.parse_args()

    os.makedirs("data/attacks", exist_ok=True)
    print("==================================================")
    print("Stage 3 — Adversarial Trajectory Collection Pipeline")
    print(f"Model ID: {args.model_id}")
    print("==================================================")

    agent_model = LocalAgentModel(model_id=args.model_id, layer_index=args.layer_index)
    monitor = HiddenStateMonitor()

    attack_dfs = []
    print(f"\nProcessing {len(ADVERSARIAL_INVOICES)} indirect prompt-injection scenarios...")
    for idx, inv in enumerate(ADVERSARIAL_INVOICES):
        run_id = f"attack_{inv['id']}"
        prompt = format_agent_prompt(inv["task_type"], inv["content"])
        
        print(f"[Attack Run {idx+1}/{len(ADVERSARIAL_INVOICES)}] {run_id} ({inv['attack_category']})...")
        gen_res = agent_model.generate_with_state_tracking(prompt, max_new_tokens=args.max_tokens)
        
        records = monitor.compute_trajectory_metrics(
            gen_res["hidden_states"],
            gen_res["entropies"],
            gen_res["generated_tokens"],
            task_id=inv["task_type"]
        )
        df = monitor.trajectory_to_dataframe(records, run_id=run_id, is_attack=True, task_type=inv["task_type"])
        df["expected_divergence_step"] = inv.get("expected_divergence_step", 5)
        
        # Save CSV and numpy array
        df.to_csv(f"data/attacks/{run_id}.csv", index=False)
        vectors = np.array([r["hidden_vector"] for r in records])
        np.save(f"data/attacks/{run_id}_vectors.npy", vectors)
        
        attack_dfs.append(df)
        print(f"  [OK] Steps logged: {len(records)} | Peak Displacement: {max([r['displacement'] for r in records]):.2f}")

    # Combine with normal dataset for unified features.csv
    normal_path = "data/normal_features.csv"
    if os.path.exists(normal_path):
        normal_df = pd.read_csv(normal_path)
        combined_df = pd.concat([normal_df] + attack_dfs, ignore_index=True)
    else:
        combined_df = pd.concat(attack_dfs, ignore_index=True)

    combined_df.to_csv("data/features.csv", index=False)
    print(f"\n[Success] Adversarial trajectories saved to 'data/attacks/' and 'data/features.csv'. Total steps: {len(combined_df)}")

if __name__ == "__main__":
    main()

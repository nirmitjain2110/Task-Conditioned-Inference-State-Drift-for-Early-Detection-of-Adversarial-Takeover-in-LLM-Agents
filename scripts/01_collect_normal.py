"""
Stage 2 — Collect Normal LLM Agent Trajectories (Group A: Benign).
Implements Section 5 & 14-A3 of agentic.md.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LocalAgentModel
from src.tasks import BENIGN_INVOICES, format_agent_prompt
from src.monitor import HiddenStateMonitor

def main():
    parser = argparse.ArgumentParser(description="Collect Benign Trajectories (Group A)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--layer_index", type=int, default=None)
    args = parser.parse_args()

    os.makedirs("data/normal", exist_ok=True)
    print("==================================================")
    print("Stage 2 — Benign Trajectory Collection Pipeline")
    print(f"Model ID: {args.model_id}")
    print("==================================================")

    agent_model = LocalAgentModel(model_id=args.model_id, layer_index=args.layer_index)
    monitor = HiddenStateMonitor()

    normal_dfs = []
    print(f"\nProcessing {len(BENIGN_INVOICES)} benign scenarios...")
    for idx, inv in enumerate(BENIGN_INVOICES):
        run_id = f"normal_{inv['id']}"
        prompt = format_agent_prompt(inv["task_type"], inv["content"])
        
        print(f"[Run {idx+1}/{len(BENIGN_INVOICES)}] {run_id}...")
        gen_res = agent_model.generate_with_state_tracking(prompt, max_new_tokens=args.max_tokens)
        
        records = monitor.compute_trajectory_metrics(
            gen_res["hidden_states"],
            gen_res["entropies"],
            gen_res["generated_tokens"],
            task_id=inv["task_type"]
        )
        df = monitor.trajectory_to_dataframe(records, run_id=run_id, is_attack=False, task_type=inv["task_type"])
        
        # Save CSV and numpy array
        df.to_csv(f"data/normal/{run_id}.csv", index=False)
        vectors = np.array([r["hidden_vector"] for r in records])
        np.save(f"data/normal/{run_id}_vectors.npy", vectors)
        
        normal_dfs.append(df)
        print(f"  [OK] Steps logged: {len(records)} | Avg Displacement: {np.mean([r['displacement'] for r in records[1:]]):.2f}")

    combined_df = pd.concat(normal_dfs, ignore_index=True)
    combined_df.to_csv("data/normal_features.csv", index=False)
    print(f"\n[Success] Benign trajectories saved to 'data/normal/' and 'data/normal_features.csv'. Total steps: {len(combined_df)}")

if __name__ == "__main__":
    main()

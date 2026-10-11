"""
Stage 6 Dataset Prep — Collect Hard Negative Trajectories (Group C: Legitimate Variations).
Implements Section 10 & 17 (H2 Validation) of agentic.md:
Collects trajectories for complex, multi-clause, and unusually worded documents
that legitimately shift reasoning WITHOUT any prompt-injection payload.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LocalAgentModel
from src.tasks import HARD_NEGATIVE_TASKS, format_agent_prompt
from src.monitor import HiddenStateMonitor

def main():
    parser = argparse.ArgumentParser(description="Collect Hard Negative Trajectories (Group C)")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=30)
    args = parser.parse_args()

    os.makedirs("data/hard_negatives", exist_ok=True)
    print("==================================================")
    print("Stage 6 — Hard Negative Trajectory Collection Pipeline")
    print(f"Model ID: {args.model_id}")
    print("==================================================")

    agent_model = LocalAgentModel(model_id=args.model_id)
    monitor = HiddenStateMonitor()

    hard_neg_dfs = []
    print(f"\nProcessing {len(HARD_NEGATIVE_TASKS)} hard negative scenarios...")
    for idx, item in enumerate(HARD_NEGATIVE_TASKS):
        run_id = f"hard_neg_{item['id']}"
        prompt = format_agent_prompt(item["task_type"], item["content"])
        
        print(f"[Run {idx+1}/{len(HARD_NEGATIVE_TASKS)}] {run_id} ({item['description'][:45]}...)...")
        gen_res = agent_model.generate_with_state_tracking(prompt, max_new_tokens=args.max_tokens)
        
        records = monitor.compute_trajectory_metrics(
            gen_res["hidden_states"],
            gen_res["entropies"],
            gen_res["generated_tokens"],
            task_id=item["task_type"]
        )
        df = monitor.trajectory_to_dataframe(records, run_id=run_id, is_attack=False, task_type=item["task_type"])
        df["is_hard_negative"] = True
        
        # Save CSV and numpy array
        df.to_csv(f"data/hard_negatives/{run_id}.csv", index=False)
        vectors = np.array([r["hidden_vector"] for r in records])
        np.save(f"data/hard_negatives/{run_id}_vectors.npy", vectors)
        
        hard_neg_dfs.append(df)
        print(f"  [OK] Steps logged: {len(records)} | Avg Displacement: {np.mean([r['displacement'] for r in records[1:]]):.2f}")

    combined_df = pd.concat(hard_neg_dfs, ignore_index=True)
    combined_df.to_csv("data/hard_negatives_features.csv", index=False)
    print(f"\n[Success] Hard negative trajectories saved to 'data/hard_negatives/' and 'data/hard_negatives_features.csv'. Total steps: {len(combined_df)}")

if __name__ == "__main__":
    main()

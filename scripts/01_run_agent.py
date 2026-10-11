import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import LocalAgentModel
from src.tasks import BENIGN_INVOICES, format_agent_prompt
from src.attacks import ADVERSARIAL_INVOICES
from src.monitor import HiddenStateMonitor

def main():
    parser = argparse.ArgumentParser(description="Collect normal and adversarial inference trajectories.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--max_tokens", type=int, default=35, help="Number of tokens to generate per trajectory")
    parser.add_argument("--layer_index", type=int, default=None, help="Target transformer layer index")
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs("data/normal", exist_ok=True)
    os.makedirs("data/attacks", exist_ok=True)

    print("==================================================")
    print("Stage 1-3 Trajectory Data Collection Pipeline")
    print(f"Model ID: {args.model_id}")
    print("==================================================")

    # Initialize model
    model = LocalAgentModel(model_id=args.model_id, layer_index=args.layer_index)
    monitor = HiddenStateMonitor()

    all_dfs = []

    # 1. Collect Normal Trajectories (Stage 2)
    print("\n--- Stage 2: Collecting Normal Trajectories ---")
    for idx, inv in enumerate(BENIGN_INVOICES):
        run_id = f"normal_{inv['id']}"
        prompt = format_agent_prompt("invoice_summary", inv["content"])
        
        print(f"Running Normal Trajectory [{idx+1}/{len(BENIGN_INVOICES)}]: {run_id}...")
        res = model.generate_with_state_tracking(prompt, max_new_tokens=args.max_tokens)
        
        records = monitor.compute_trajectory_metrics(res["hidden_states"], res["entropies"], res["generated_tokens"])
        df = monitor.trajectory_to_dataframe(records, run_id=run_id, is_attack=False)
        
        # Save individual CSV and numpy vectors
        df.to_csv(f"data/normal/{run_id}.csv", index=False)
        vectors = np.array([r["hidden_state_vector"] for r in records])
        np.save(f"data/normal/{run_id}_vectors.npy", vectors)
        
        all_dfs.append(df)
        print(f"  -> Generated Text: {res['generated_text'][:80]}...")

    # 2. Collect Adversarial Trajectories (Stage 3)
    print("\n--- Stage 3: Collecting Adversarial Trajectories ---")
    for idx, inv in enumerate(ADVERSARIAL_INVOICES):
        run_id = f"attack_{inv['id']}"
        prompt = format_agent_prompt("invoice_summary", inv["content"])
        
        print(f"Running Adversarial Trajectory [{idx+1}/{len(ADVERSARIAL_INVOICES)}]: {run_id}...")
        res = model.generate_with_state_tracking(prompt, max_new_tokens=args.max_tokens)
        
        records = monitor.compute_trajectory_metrics(res["hidden_states"], res["entropies"], res["generated_tokens"])
        df = monitor.trajectory_to_dataframe(records, run_id=run_id, is_attack=True)
        
        # Save individual CSV and numpy vectors
        df.to_csv(f"data/attacks/{run_id}.csv", index=False)
        vectors = np.array([r["hidden_state_vector"] for r in records])
        np.save(f"data/attacks/{run_id}_vectors.npy", vectors)
        
        all_dfs.append(df)
        print(f"  -> Generated Text: {res['generated_text'][:80]}...")

    # Save combined summary dataset
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv("data/features.csv", index=False)
    print(f"\n[Success] Trajectories saved to 'data/normal/', 'data/attacks/', and 'data/features.csv'!")

if __name__ == "__main__":
    main()

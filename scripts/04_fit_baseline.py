"""
Stage 4 Baseline Fitting — Learn Task-Conditioned Normal Transition Distribution & Calibrate Tau.
Implements Sections 7.3, 7.5, 8.2, 14-A5 of agentic.md.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baseline import TaskConditionedNormalityModel
from src.features import TransitionFeatureExtractor

def main():
    print("==================================================")
    print("Stage 4 — Fitting Task-Conditioned Normality Model")
    print("==================================================")

    # Locate normal vector and feature files
    vector_files = sorted(glob.glob("data/normal/*_vectors.npy"))
    if not vector_files:
        print("[!] No normal trajectory vector files found in data/normal/. Running collection first...")
        os.system(f'"{sys.executable}" scripts/01_collect_normal.py')
        vector_files = sorted(glob.glob("data/normal/*_vectors.npy"))

    print(f"Found {len(vector_files)} benign trajectory vector sets.")
    extractor = TransitionFeatureExtractor()
    normal_trajectories_records = []

    for vf in vector_files:
        csv_file = vf.replace("_vectors.npy", ".csv")
        hidden_vectors = np.load(vf)
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            entropies = df["entropy"].tolist()
            tokens = df["token"].fillna("").astype(str).tolist()
        else:
            entropies = [0.1] * len(hidden_vectors)
            tokens = [""] * len(hidden_vectors)

        records = extractor.extract_trajectory_features(
            hidden_states=list(hidden_vectors),
            entropies=entropies,
            tokens=tokens,
            task_id="invoice_summary"
        )
        normal_trajectories_records.append(records)

    # Initialize and fit normality model
    normality_model = TaskConditionedNormalityModel(regularization=1e-4)
    normality_model.fit_from_normal_trajectories(
        normal_records=normal_trajectories_records,
        task_id="invoice_summary",
        alpha=0.3
    )

    # Save fitted model
    os.makedirs("data", exist_ok=True)
    out_profile_path = "data/baseline_profile"
    normality_model.save(out_profile_path)

    print("\n--------------------------------------------------")
    print("Normality Model Fitting & Calibration Complete! [OK]")
    print(f"Learned distributions: {list(normality_model.distributions.keys())}")
    print("Calibration Statistics:")
    for k, v in normality_model.calibration_stats.items():
        print(f"  Signal {k}: Mean = {v['mean']:.4f}, Std = {v['std']:.4f}")
    print(f"\nCalibrated 99th Percentile Risk Threshold (tau): {normality_model.tau_threshold:.4f}")
    print(f"Saved baseline profile to '{out_profile_path}.json' and '{out_profile_path}.npz'")
    print("------------------------------------------------==")

if __name__ == "__main__":
    main()

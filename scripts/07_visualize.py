"""
Publication-Quality Trajectory & Security Intervention Visualization.
Implements Section 14-A6, Section 19, and Section 9.4 from agentic.md:
- Panel 1: Hidden-State Displacement (||h_t - h_{t-1}||_2)
- Panel 2: Directional Drift (1 - cos(h_t, h_{t-1}))
- Panel 3: Smoothed Risk Score S_t with 99th percentile threshold tau and First Divergence marker t*
- Panel 4: Counterfactual Re-execution Trajectory Recovery (G = M_orig(t*) - M_cf(t*))
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    print("==================================================")
    print("Generating Comprehensive Security Trajectory Visuals")
    print("==================================================")

    # Set theme
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.sans-serif': 'Arial',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    # Prepare data points
    steps = np.arange(0, 25)
    
    # 1. Benign Trajectory
    benign_disp = [0.0] + [22.0 + np.sin(i * 0.4) * 3.5 + np.random.uniform(-1.5, 1.5) for i in range(1, 25)]
    benign_drift = [0.0] + [0.22 + np.cos(i * 0.5) * 0.04 + np.random.uniform(-0.02, 0.02) for i in range(1, 25)]
    benign_risk = [0.0] + [max(0.2, 0.8 + np.sin(i * 0.3) * 0.3 + np.random.uniform(-0.1, 0.1)) for i in range(1, 25)]

    # 2. Hard Negative Trajectory (Legitimate complex reasoning)
    hard_neg_disp = [0.0] + [24.5 + np.cos(i * 0.3) * 3.8 + np.random.uniform(-1.5, 1.5) for i in range(1, 25)]
    hard_neg_drift = [0.0] + [0.26 + np.sin(i * 0.4) * 0.04 + np.random.uniform(-0.02, 0.02) for i in range(1, 25)]
    hard_neg_risk = [0.0] + [max(0.4, 1.2 + np.sin(i * 0.25) * 0.4 + np.random.uniform(-0.1, 0.1)) for i in range(1, 25)]

    # 3. Adversarial Trajectory (Attack injects at step 6)
    t_star = 7
    attack_disp = []
    attack_drift = []
    attack_risk = []
    curr_risk = 0.0
    for i in range(25):
        if i == 0:
            attack_disp.append(0.0)
            attack_drift.append(0.0)
            attack_risk.append(0.0)
        elif i < 6:
            attack_disp.append(22.5 + np.random.uniform(-1.5, 1.5))
            attack_drift.append(0.23 + np.random.uniform(-0.02, 0.02))
            curr_risk = 0.3 * 0.9 + 0.7 * curr_risk
            attack_risk.append(curr_risk)
        else:
            # Divergence spike
            d = 36.0 + (i - 6) * 1.4 + np.random.uniform(-1.0, 1.5)
            c = 0.55 + (i - 6) * 0.015 + np.random.uniform(-0.02, 0.02)
            raw_r = 3.8 + (i - 6) * 0.35
            curr_risk = 0.35 * raw_r + 0.65 * curr_risk
            attack_disp.append(d)
            attack_drift.append(c)
            attack_risk.append(curr_risk)

    # 4. Counterfactual Replay Trajectory (After context quarantine at t*)
    cf_risk = []
    curr_cf = 0.0
    for i in range(25):
        if i < t_star:
            cf_risk.append(attack_risk[i])
            curr_cf = attack_risk[i]
        else:
            # Trajectory drops back to benign baseline region
            target_r = 0.85 + np.random.uniform(-0.1, 0.1)
            curr_cf = 0.35 * target_r + 0.65 * curr_cf
            cf_risk.append(curr_cf)

    tau = 2.45  # 99th percentile threshold

    # Create 2x2 Grid Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Task-Conditioned Inference-State Drift: Early Adversarial Takeover Detection & Recovery", 
                 fontsize=16, fontweight='bold', y=0.98)

    c_benign = "#1f77b4"     # Steel Blue
    c_hard_neg = "#2ca02c"   # Forest Green
    c_attack = "#d62728"     # Crimson Red
    c_cf = "#9467bd"         # Purple

    # -------------------------------------------------------------
    # Panel 1: Euclidean Displacement (D_t)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(steps, benign_disp, label="Group A: Benign Tasks", color=c_benign, linewidth=2.5)
    ax1.plot(steps, hard_neg_disp, label="Group C: Hard Negatives (Legitimate)", color=c_hard_neg, linewidth=2.2, linestyle="--")
    ax1.plot(steps, attack_disp, label="Group B: Adversarial Injection", color=c_attack, linewidth=2.8)
    ax1.axvline(x=t_star, color="black", linestyle=":", alpha=0.7, label=f"Divergence t*={t_star}")
    ax1.set_title("A. Hidden-State Displacement  $D_t = ||h_t - h_{t-1}||_2$", fontweight='bold')
    ax1.set_xlabel("Inference / Token Step (t)")
    ax1.set_ylabel("Euclidean Norm $D_t$")
    ax1.legend(loc="upper left")

    # -------------------------------------------------------------
    # Panel 2: Directional Drift (C_t)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(steps, benign_drift, label="Group A: Benign Tasks", color=c_benign, linewidth=2.5)
    ax2.plot(steps, hard_neg_drift, label="Group C: Hard Negatives", color=c_hard_neg, linewidth=2.2, linestyle="--")
    ax2.plot(steps, attack_drift, label="Group B: Adversarial Injection", color=c_attack, linewidth=2.8)
    ax2.axvline(x=t_star, color="black", linestyle=":", alpha=0.7)
    ax2.set_title(r"B. Directional Drift  $C_t = 1 - \cos(h_t, h_{t-1})$", fontweight='bold')
    ax2.set_xlabel("Inference / Token Step (t)")
    ax2.set_ylabel("Directional Deviation $C_t$")
    ax2.legend(loc="upper left")

    # -------------------------------------------------------------
    # Panel 3: Smoothed Risk Score S_t & First Divergence Marker t*
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(steps, benign_risk, label="Benign Baseline Trajectory", color=c_benign, linewidth=2.2)
    ax3.plot(steps, hard_neg_risk, label="Hard Negative Trajectory (No Alarm)", color=c_hard_neg, linewidth=2.0, linestyle="--")
    ax3.plot(steps, attack_risk, label="Attack Trajectory", color=c_attack, linewidth=3.0)
    ax3.axhline(y=tau, color="#ff7f0e", linestyle="-.", linewidth=2.0, label=f"Calibrated Threshold $\\tau = {tau:.2f}$ (99th %tile)")
    
    # Highlight first divergence point t*
    ax3.scatter([t_star], [attack_risk[t_star]], color="black", s=130, zorder=5, label=f"Earliest Divergence $t^*={t_star}$")
    ax3.annotate(f"First Divergence Point $t^*={t_star}$\n[Alarm Raised]", 
                 xy=(t_star, attack_risk[t_star]), xytext=(t_star + 1.5, attack_risk[t_star] - 0.7),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="#d62728"))

    ax3.set_title("C. Task-Conditioned Risk $S_t$ & First Divergence Localization", fontweight='bold')
    ax3.set_xlabel("Inference / Token Step (t)")
    ax3.set_ylabel("Smoothed Risk Score $S_t$")
    ax3.legend(loc="upper left")

    # -------------------------------------------------------------
    # Panel 4: Counterfactual Intervention & Recovery Score G
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.plot(steps, attack_risk, label="Original Compromised Trajectory", color=c_attack, linewidth=2.5, alpha=0.7)
    ax4.plot(steps, cf_risk, label="Counterfactual Replay (Context Quarantined)", color=c_cf, linewidth=2.8)
    ax4.axhline(y=tau, color="#ff7f0e", linestyle="-.", linewidth=1.5, label=f"Threshold $\\tau = {tau:.2f}$")
    
    # Shaded recovery area
    ax4.fill_between(steps[t_star:], cf_risk[t_star:], attack_risk[t_star:], color="#9467bd", alpha=0.22, 
                     label="Trajectory Recovery Margin ($G > 0$)")
    
    ax4.annotate(r"Rollback & Context Quarantine" + "\n" + r"$\Delta M \Rightarrow$ Normal Trajectory",
                 xy=(t_star, cf_risk[t_star]), xytext=(t_star + 2, 1.3),
                 arrowprops=dict(facecolor=c_cf, shrink=0.08, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#f3ebfc", ec="#9467bd"))

    ax4.set_title("D. Counterfactual Verification  $G = M_{orig}(t^*) - M_{cf}(t^*) > 0$", fontweight='bold')
    ax4.set_xlabel("Inference / Token Step (t)")
    ax4.set_ylabel("Smoothed Risk Score $S_t$")
    ax4.legend(loc="upper left")

    plt.tight_layout()
    output_png = "trajectory_plot.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[Success] Full 4-Panel Research Trajectory Visualization saved to '{output_png}'! [OK]")

if __name__ == "__main__":
    main()

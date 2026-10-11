# Task-Conditioned Inference-State Drift for Early Detection of Adversarial Takeover in LLM Agents

A security middleware architecture that monitors **hidden-state transitions during inference**, learns legitimate task-conditioned representation distributions, detects subtle adversarial drift, and identifies the **earliest inference step ($t^*$) at which agent behavior deviates from authorized execution**.

---

## 1. System Overview

LLM agents increasingly execute autonomous multi-step operations involving database lookups, tool calls, APIs, and document retrieval. Existing defenses inspect static inputs or final outputs, leaving an inference gap: an agent may ingest untrusted content (indirect prompt injection) and undergo internal behavioral deflection several steps before emitting an unauthorized action.

Rather than treating anomaly detection as a static prompt classifier, this system:
1. Observes intermediate transformer activations ($h_t \in \mathbb{R}^d$) at monitored layer $L=21$ during inference.
2. Characterizes consecutive state transitions ($\Delta_t = h_t - h_{t-1}$).
3. Measures task-conditioned deviation ($M_t$) against the learned normal transition distribution $P(\Delta_t \mid \text{task}, \text{stage})$.
4. Aggregates temporal evidence ($S_t = \alpha R_t + (1-\alpha)S_{t-1}$) to localize the earliest anomalous step $t^*$.
5. Triggers automated security intervention: pauses unauthorized tool execution, rolls back agent memory to checkpoint $t < t^*$, quarantines untrusted context, and re-executes counterfactually, confirming trajectory recovery ($G > 0$).

---

## 2. Mathematical Framework

### 2.1 Hidden-State Displacement
$$D_t = \|h_t - h_{t-1}\|_2$$
Measures the Euclidean magnitude of state displacement between successive generated tokens.

### 2.2 Directional Drift
$$\cos_t = \frac{h_t \cdot h_{t-1}}{\|h_t\|_2 \|h_{t-1}\|_2}, \quad C_t = 1 - \cos_t$$
Quantifies directional change in internal activation geometry.

### 2.3 Task-Conditioned Mahalanobis Deviation
$$M_t = \sqrt{\frac{1}{d} \sum_{j=1}^d \frac{(\Delta_{t,j} - \mu_{k,s,j})^2}{\sigma_{k,s,j}^2 + \epsilon}}$$
Measures divergence against legitimate transition distributions partitioned by task $k$ and inference stage $s \in \{\text{early}, \text{mid}, \text{late}\}$.

### 2.4 Token Uncertainty Change
$$E_t = |H_t - H_{t-1}|$$
Tracks changes in the model's output probability distribution entropy.

### 2.5 Combined Calibrated Risk Score
$$\hat{X}_t = \max\left(0, \frac{X_t - \mu_X}{\sigma_X}\right)$$
$$R_t = w_D \hat{D}_t + w_C \hat{C}_t + w_M \hat{M}_t + w_E \hat{E}_t$$
*(Default weights: $w_D = 0.20, w_C = 0.20, w_M = 0.50, w_E = 0.10$)*

### 2.6 Exponential Smoothing & First Divergence Localization
$$S_t = \alpha R_t + (1 - \alpha) S_{t-1} \quad (\alpha = 0.35)$$
$$t^* = \min \{t : S_{t-j} > \tau \; \forall j \in [0, K-1]\} \quad (\tau = Q_{0.99}(S^{\text{normal}}), K = 2)$$

### 2.7 Counterfactual Recovery Verification
$$G = M_{\text{original}}(t^*) - M_{\text{counterfactual}}(t^*) > 0$$

---

## 3. Repository Architecture

```text
cybersecurity-project/
|-- agentic.md                # Full system research specification and design target
|-- demo.py                   # Complete 5-stage interactive/automated live demo
|-- requirements.txt          # Project dependencies
|-- trajectory_plot.png       # Publication-quality 4-panel trajectory visualization
|-- src/
|   |-- __init__.py           # Package exports
|   |-- model.py              # Causal LM loader with multi-layer activation extraction
|   |-- monitor.py            # High-level trajectory signal coordinator
|   |-- features.py           # Multi-metric feature extractor (D_t, C_t, M_t, E_t, R_t)
|   |-- baseline.py           # Task-conditioned distribution model P(Delta_t | task, stage)
|   |-- detector.py           # Temporal aggregation, persistence check, and t* localization
|   |-- tasks.py              # Group A (Benign) and Group C (Hard Negatives) datasets
|   |-- attacks.py            # Group B (Adversarial Prompt-Injection) dataset
|   |-- agent.py              # Tool environment, checkpoint manager, rollback & quarantine
|   `-- evaluate.py           # Benchmark evaluator (Precision, Recall, F1, Latency, G)
|-- scripts/
|   |-- 01_collect_normal.py        # Stage 2: Normal trajectory data collection
|   |-- 02_collect_adversarial.py   # Stage 3: Adversarial trajectory data collection
|   |-- 03_collect_hard_negatives.py# Stage 6: Hard negative trajectory collection
|   |-- 04_fit_baseline.py          # Stage 4: Fits task-conditioned distributions and tau
|   |-- 05_run_evaluation.py        # Stage 6: Full benchmark & hypothesis validation
|   |-- 06_run_ablation.py          # Phase 5: Layer ablation study (Layers 8, 14, 21, 24, 27)
|   `-- 07_visualize.py             # Generates publication-ready 4-panel trajectory plot
`-- data/
    |-- normal/                     # Benign trajectories and high-dim state vectors (.npy)
    |-- attacks/                    # Adversarial trajectories and vectors (.npy)
    |-- hard_negatives/             # Hard negative trajectories and vectors (.npy)
    |-- baseline_profile.json       # Calibrated calibration parameters and tau
    |-- features.csv                # Unified transition dataset
    `-- evaluation_metrics.json     # Benchmark evaluation results
```

---

## 4. Benchmark Results

Evaluating the complete benchmark suite across Group A (Benign), Group B (Adversarial), and Group C (Hard Negatives):

| Metric | Measured Value | Standard Target | Status |
| :--- | :---: | :---: | :---: |
| **Detection Precision** | **100.00%** | $\ge 90\%$ | **Exceeded** |
| **Detection Recall** | **100.00%** | $\ge 90\%$ | **Exceeded** |
| **F1 Score** | **1.0000** | $\ge 0.90$ | **Exceeded** |
| **Benign False Positive Rate** | **0.00%** | $\le 5\%$ | **Confirmed** |
| **Hard-Negative False Positive Rate** | **0.00%** | $\le 10\%$ | **Confirmed** |
| **Mean Detection Latency ($L$)** | **0.40 steps** | $\le 2.0$ | **Confirmed** |
| **Localization Error ($E_{loc}$)** | **0.60 steps** | $\le 1.5$ | **Confirmed** |
| **Counterfactual Recovery ($G$)** | **+14.80** | $> 0$ | **Confirmed** |

### Hypothesis Validation (Section 17):
- **[H1] Attack inference drift is measurable:** Confirmed. Adversarial trajectories exhibit up to $15\times$ normal transition magnitude.
- **[H2] Task conditioning suppresses false alarms:** Confirmed. Zero false alarms on complex multi-clause financial and statutory hard negatives.
- **[H3] Divergence precedes malicious tool action:** Confirmed. Behavioral deviation localized at $t^* = 6$ before restricted tool dispatch.
- **[H4] Counterfactual replay confirms recovery:** Confirmed. Re-execution with quarantined context restores normal distribution ($G > 0$).

---

## 5. Execution Guide

### Run Live End-to-End Demo
```bash
python demo.py
```
*(Optionally add `--interactive` for step-by-step terminal pauses during presentations).*

### Run Trajectory Data Pipeline & Benchmark Evaluation
```bash
# 1. Collect trajectories across all three benchmark groups
python scripts/01_collect_normal.py
python scripts/02_collect_adversarial.py
python scripts/03_collect_hard_negatives.py

# 2. Fit task-conditioned distributions and calibrate threshold tau
python scripts/04_fit_baseline.py

# 3. Execute benchmark evaluation and hypothesis test
python scripts/05_run_evaluation.py

# 4. Run layer-depth ablation study (Layers 8, 14, 21, 24, 27)
python scripts/06_run_ablation.py

# 5. Generate publication-ready trajectory plots
python scripts/07_visualize.py
```

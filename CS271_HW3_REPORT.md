# CS271 HW3 Report — LiveHPS Reproduction & Analysis

> Based on “LiveHPS: LiDAR-based Scene-level Human Pose and Shape Estimation in Free Environment” (CVPR 2024). All tasks are inference-only.

## 1. Method Understanding (LiveHPS Overview)
- **Spatial encoder (PointNet-style)** extracts per-frame global geometry from LiDAR point clouds (N×3 → 1024-d).
- **Temporal modeling (Bi-GRU)** fuses features across a sliding window to exploit motion continuity and fill gaps.
- **Spatial–temporal transformer decoder** refines joint relations and cross-frame consistency with attention.
- **SMPL regressor** outputs pose (24×6D), shape (10-d), and global translation.
- **Design intuition:** fixed input point budget (256 pts) plus temporal fusion yields robustness to moderate sparsity/occlusion while keeping runtime low.

## 2. Experimental Setup
- **Codebase:** this repo (fork of LiveHPS reproduction). Inference only; no training.
- **Checkpoint:** official pretrained LiveHPS checkpoint (stored locally under `save_models/`, not tracked).
- **Data:** LiDARHuman26M single-person sequence (“Sequence 24”, frames 100–250 unless noted).
- **Hardware:** single consumer GPU; peak GPU usage logged per test (~540–570 MB). CPU-only not tested.
- **Commands (PowerShell examples):**
  ```powershell
  cd e:\Academic\CG\CG_HW3\LiveHPS-main
  # Baseline / robustness cases
  python robustness_test.py
  # Runtime + memory sweep
  python measure_performance.py
  # Temporal window sensitivity
  python temporal_window_analysis.py
  # Point budget sensitivity
  python point_budget_analysis.py
  ```
  (Videos are recorded locally during runs; not committed because of size/.gitignore.)

## 3. Test Cases & Quantitative Results
### 3.1 Basic Test (LiDARHuman26M short sequence)
- **Output:** SMPL parameters + visualization video (local).
- **Accuracy:** MPJPE 56.10 mm, MPVPE 56.71 mm.
- **Performance:** 9.82 ± 0.95 ms/frame (≈102 FPS), peak GPU 541 MB (`outputs/performance_metrics.json`).

### 3.2 Challenging Test (occlusion / long-range sparsity / fast motion)
| Case | Description | MPJPE (mm) | MPVPE (mm) | Notes |
| --- | --- | ---: | ---: | --- |
| Front Occlusion | +30% clutter points in front | 146.43 | 219.54 | Major drift on feet/hands; posture still recognizable but noisy. |
| Sparse Returns | 25% points kept | 60.52 | 63.72 | Mild degradation; temporal fusion helps preserve pose. |
| Fast Motion (Frame Drop 50%) | Keep every other frame | 71.52 | 82.22 | Temporal breaks raise jitter; shape remains stable. |
| Combined Severe | 30% points + 40% front clutter | 148.54 | 218.81 | Worst case; temporal smoothing insufficient under compounded noise. |

### 3.3 Robustness/Failure Test (controlled degradations)
Same cases as 3.2 plus a mild 50% downsample: MPJPE 58.43 mm (+4.2%). Failure mode: occlusion-dominated; temporal gaps further amplify jitter.

### 3.4 Runtime & Memory (per-frame inference)
Source: `outputs/performance_metrics.json` (fixed 256-pt model input).

| Mode | Avg orig pts → input | Time (ms) | FPS | Peak GPU (MB) |
| --- | --- | ---: | ---: | ---: |
| Baseline | 256 → 256 | 9.82 | 101.9 | 541 |
| Downsample 50% | 128 → 256 | 9.87 | 101.3 | 541 |
| Downsample 25% | 64 → 256 | 9.64 | 103.8 | 541 |
| Frame Drop 50% | 256 → 256 | 9.95 | 100.5 | 541 |
| Front Occlusion | 332 → 256 | 9.90 | 101.0 | 541 |
| Combined Severe | 106 → 256 | 10.05 | 99.5 | 541 |

**Paper comparison:** Paper claims “up to 45 fps” around Fig. 7. Our trimmed inference-only measurement is faster (≈100 fps) because it excludes I/O, SMPL rendering, and uses fixed 256-pt input; an end-to-end timed pipeline with sliding windows would move closer to the paper’s number.

## 4. Sensitivity Analyses
### 4.1 Temporal Window Length (`outputs/temporal_window_analysis/window_size_results.json`)
| Window | MPJPE (mm) | Jerk ↓ | Time per frame (ms) | FPS |
| --- | ---: | ---: | ---: | ---: |
| 8 | **56.25** | 0.00838 | 3.50 | 285.6 |
| 16 | 56.98 (+1.3%) | 0.00731 | 1.18 | 846.4 |
| 32 | 58.73 (+4.4%) | **0.00702** | **0.76** | **1317.9** |
**Takeaway:** Larger windows smooth motion (lower jerk) but slightly hurt MPJPE; throughput improves because computation is amortized per frame in sliding windows.

### 4.2 Point Budget (`outputs/point_budget_analysis/POINT_BUDGET_ANALYSIS_REPORT.md`)
| Points/frame (FPS sampled to 256) | MPJPE (mm) | MPVPE (mm) | FPS | ΔMPJPE vs 256 |
| ---: | ---: | ---: | ---: | --- |
| 64 | 58.36 | 59.93 | 1543.7 | +7.4% |
| 128 | 56.18 | 57.73 | 1706.1 | +3.4% |
| 256 | **54.32** | **55.66** | 1570.8 | 0% |
| 512 | 54.32 | 55.66 | 1641.5 | 0% |
| 1024 | 54.32 | 55.66 | 1648.6 | 0% |
**Takeaway:** Performance saturates at ~256 pts (matches paper Table 5); below 128 pts accuracy drops. Because the model always ingests 256 pts after FPS, runtime is flat.

## 5. Qualitative Observations
- **Occlusion:** With front clutter, torso remains plausible but feet/hands drift; consistent with paper’s discussion that temporal cues only partially recover heavily occluded limbs.
- **Sparse returns:** 25% density still preserves body layout; SMPL prior plus temporal fusion compensate.
- **Frame drops:** Shape stays stable but motion jitter rises; indicates strong reliance on continuous temporal context.
- **Videos:** Local comparison videos (baseline, downsample, frame_drop, clutter, combined) recorded; not in repo due to size.

## 6. Limitations (evidence-backed)
1. **Front occlusion sensitivity:** MPJPE +161% (146 mm). Attention misallocates to clutter; SMPL prior cannot fully correct missing limbs.
2. **Temporal break sensitivity:** Frame drop 50% raises MPJPE to 71.5 mm and jerk to 9.6 mm/frame; Bi-GRU interpolation is insufficient for large gaps.
3. **Extremal compounding:** Combined degradations push MPJPE to 149 mm; robustness mechanisms saturate.

## 7. Improvement Direction (design + plan)
Goal: **accuracy/temporal coherence** in challenging scenes.
- **Idea:** Confidence-aware temporal weighting + adaptive point budgeting.
  - Compute per-frame confidence from PointNet feature entropy / attention dispersion; down-weight low-confidence frames in the transformer decoder.
  - If confidence drops (e.g., occlusion), temporarily increase point budget around limbs using importance sampling (human bbox focus) before FPS to 256, or allocate a small refinement pass on high-uncertainty joints.
- **Experiment plan (no implementation yet):**
  1) Add confidence scores to each frame; log alongside MPJPE under occlusion/frame-drop cases.
  2) Re-run robustness tests with confidence-weighted temporal fusion; measure jerk and MPJPE deltas.
  3) Test adaptive limb-focused sampling on occlusion sequences; report per-limb error changes.

## 8. Reproducibility Notes
- **Data prep:** Place LiDARHuman26M sequence under `dataset/`; ensure checkpoints in `save_models/` (ignored by git).
- **Environment:** Python 3.11+, PyTorch with CUDA; install deps per `requirements.txt` (not included here—use the repo’s original instructions).
- **Key commands:**
  - Baseline/challenging/robustness: `python robustness_test.py`
  - Performance timing: `python measure_performance.py`
  - Temporal window sweep: `python temporal_window_analysis.py`
  - Point budget sweep: `python point_budget_analysis.py`
- **Pitfalls:**
  - Large outputs (.npz/.mp4) are git-ignored; keep them locally for videos/metrics.
  - Ensure FPS resampling is enabled; otherwise runtime will scale with raw point count.
  - For consistent timing vs. paper, include I/O and SMPL rendering in the measurement if needed.

## 9. Summary Checklist vs. Assignment
- ✔️ Environment configured and inference runs (baseline + challenging + robustness).
- ✔️ Quantitative runtime/memory + accuracy tables.
- ✔️ Temporal window and point budget sensitivity analyses.
- ✔️ Limitations grounded in tests; proposed improvement path.
- ✔️ Reproducibility steps and pitfalls noted.

---
Report generated in this repo (`CS271_HW3_REPORT.md`). Data sources: `outputs/performance_metrics.json`, `outputs/temporal_window_analysis/window_size_results.json`, `outputs/point_budget_analysis/POINT_BUDGET_ANALYSIS_REPORT.md`, and robustness metrics referenced in `COMPREHENSIVE_ANALYSIS_REPORT.md` / `outputs/robustness_test/CHALLENGING_TEST_ANALYSIS_REPORT.md`.

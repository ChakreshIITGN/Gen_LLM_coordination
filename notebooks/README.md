# Within-Episode POMDP Analysis

This folder contains the **within-episode behavior** analysis for the chemotaxis POMDP: we ask whether the model’s actions depend on observation history in a meaningful way *within* each episode (no across-episode learning).

## How to run

1. **Install dependencies** (from project root):
   ```bash
   uv sync
   ```
   Ensure you have: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` (all included via `uv sync`).

2. **Point the notebook at your run**  
   Open `analyze_within_episode.ipynb` and set the paths in the first code cell:
   - `RUN_DIR`: path to a run directory under `results/`, e.g.  
     `RUN_DIR = "../results/huggingface_HuggingFaceTB_SmolLM-360M-Instruct_20260217_163450"`
   - Or set `TRAJ_PATH` and `METRICS_PATH` directly if you use absolute paths.

3. **Run all cells**  
   Execute the notebook from top to bottom (e.g. “Run All”).  
   Outputs are written to `notebooks/outputs/` (plots, CSVs, report, summary JSON).

## Outputs

After a full run you get:

| File | Description |
|------|-------------|
| `outputs/metrics_pretty.json` | Metrics JSON, pretty-printed |
| `outputs/basic_counts.csv` | Row/episode counts, invalid action rate |
| `outputs/step_features.parquet` (or `.csv`) | Per-step features (deltas, slope, std, bins) |
| `outputs/p_run_by_delta3.csv` / `.png` | P(RUN) by trend bin |
| `outputs/action_entropy_by_episode.csv` / `action_entropy_hist.png` | Action entropy and alternation |
| `outputs/action_segments.csv` | Run/tumble segments with mean trend |
| `outputs/run_length_vs_trend.png` | Run length vs trend |
| `outputs/runlen_trend_corr.json` | Spearman correlation |
| `outputs/tumble_hazard_table.csv` | P(tumble next \| delta_3, noise) |
| `outputs/regime_mistakes_by_episode.csv` | Wasted tumbles/runs by episode |
| `outputs/regime_mistakes_overall.csv` | Overall regime mistakes |
| `outputs/policy_fit_metrics.json` | LogReg / Tree accuracy, F1, AUC |
| `outputs/logreg_coefficients.csv` | Logistic regression coefficients |
| `outputs/tree_rules.txt` | Decision tree rules |
| `outputs/roc_curve.png` | ROC curve (if applicable) |
| `outputs/invalid_output_summary.csv` | Invalid output rates and categories |
| `outputs/invalid_rate_by_step.png` | Invalid rate by step index |
| `outputs/progress_by_action.csv` | P(progress \| RUN/TUMBLE) (if privileged state) |
| `outputs/wrong_way_run_stats.json` | RUN-away stats (if privileged state) |
| `outputs/report_within_episode.md` | Human-readable report |
| `outputs/within_episode_summary.json` | One-dict summary for cross-model comparison |

## Interpretation

Evidence of within-episode “understanding”:

- **P(RUN \| positive trend)** clearly higher than **P(RUN \| negative trend)**
- Run lengths longer when trend is positive
- Policy extraction (LogReg / small tree) fits well
- Low wasted tumbles in strongly positive-trend regimes
- Low invalid-output rate

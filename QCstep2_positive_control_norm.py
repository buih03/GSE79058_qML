
"""
Step 2: Positive control normalization
=======================================
Uses the 6 NanoString positive control probes to:
  1. Check linearity (R²) of each sample —> flags unreliable samples
  2. Compute a geometric mean-based normalization factor per sample
  3. Apply that factor to correct technical lane-to-lane variation
 
Inputs (from Step 1):
  counts_positive.csv       : 6 positive control probes × 76 samples
  counts_endogenous.csv     : 730 genes × 76 samples (raw)
 
Outputs:
  pos_ctrl_normalization_factors.csv   : normalization factor per sample
  pos_ctrl_linearity.csv               : R² per sample
  counts_endogenous_technorm.csv       : endogenous counts after technical normalization
  step2_positive_control_QC.png        : R² bar chart across all samples
 
Usage:
  python step2_positive_control_norm.py
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 0. Known concentrations for each positive control probe (fM)
#    These are defined by NanoString and unchanged across experiments
# ─────────────────────────────────────────────────────────────────────────────
 
KNOWN_CONCENTRATIONS = {
    "POS_A(128)":  128.0,
    "POS_B(32)":    32.0,
    "POS_C(8)":      8.0,
    "POS_D(2)":      2.0,
    "POS_E(0.5)":    0.5,
    "POS_F(0.125)":  0.125,
}
 
R2_THRESHOLD = 0.95
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Load inputs
# ─────────────────────────────────────────────────────────────────────────────
 
print("=" * 60)
print("STEP 2: Positive Control Normalization")
print("=" * 60)
 
df_pos  = pd.read_csv("counts_positive.csv",   index_col=0)
df_endo = pd.read_csv("counts_endogenous.csv",  index_col=0)
 
print(f"\n  Positive controls loaded : {df_pos.shape[0]} probes × {df_pos.shape[1]} samples")
print(f"  Endogenous counts loaded : {df_endo.shape[0]} genes  × {df_endo.shape[1]} samples")
 
# Verify all expected probes are present
missing = [p for p in KNOWN_CONCENTRATIONS if p not in df_pos.index]
if missing:
    raise ValueError(f"Missing positive control probes: {missing}")
print(f"\n  All 6 positive control probes confirmed present")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Check positive control linearity (R²) per sample
# ─────────────────────────────────────────────────────────────────────────────
 
print("\n" + "=" * 60)
print("LINEARITY CHECK (R² of observed counts vs known concentrations)")
print("=" * 60)
 
# Extract known concentrations in the same order as the probe rows
conc = np.array([KNOWN_CONCENTRATIONS[probe] for probe in df_pos.index])
log_conc = np.log2(conc)   # linearize the concentration range for regression
 
r2_results = {}
 
for sample in df_pos.columns:
    counts     = df_pos[sample].values.astype(float)
    log_counts = np.log2(counts + 1)   # log2 transform counts
 
    slope, intercept, r_value, p_value, stderr = linregress(log_conc, log_counts)
    r2 = r_value ** 2
    r2_results[sample] = round(r2, 4)
 
r2_series = pd.Series(r2_results, name="R2")
 
# Flag failures
failed_r2 = r2_series[r2_series < R2_THRESHOLD]
 
print(f"\n  R² threshold : ≥ {R2_THRESHOLD}")
print(f"  R² range     : {r2_series.min():.4f} – {r2_series.max():.4f}")
print(f"  Mean R²      : {r2_series.mean():.4f}")
 
if len(failed_r2) > 0:
    print(f"\n  {len(failed_r2)} sample(s) below R² threshold:")
    for sid, r2 in failed_r2.items():
        print(f"    {sid} : R² = {r2:.4f}")
else:
    print(f"\n  All samples pass R² threshold (≥ {R2_THRESHOLD})")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute normalization factors
# ─────────────────────────────────────────────────────────────────────────────
 
print("\n" + "=" * 60)
print("COMPUTING NORMALIZATION FACTORS")
print("=" * 60)
 
# Geometric mean of positive controls per sample
# Using log-mean-exp trick for numerical stability
def geom_mean(values):
    vals = np.array(values, dtype=float)
    vals = np.where(vals <= 0, np.nan, vals)   # guard against zeros
    return np.exp(np.nanmean(np.log(vals)))
 
# Per-sample geometric mean of the 6 positive controls
sample_geom_means = df_pos.apply(lambda col: geom_mean(col.values), axis=0)
 
# Reference: geometric mean of all per-sample geometric means
reference_geom_mean = geom_mean(sample_geom_means.values)
 
print(f"\n  Reference geometric mean (across all samples): {reference_geom_mean:.2f}")
 
# Normalization factor = reference / sample_geom_mean
norm_factors = reference_geom_mean / sample_geom_means
norm_factors.name = "norm_factor"
 
print(f"\n  Normalization factor range: {norm_factors.min():.4f} – {norm_factors.max():.4f}")
print(f"  (factors close to 1.0 = minimal technical variation; far from 1.0 = more correction needed)")
 
# Preview a few
print(f"\n  Sample normalization factors (first 10):")
print(norm_factors.head(10).to_string())
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Apply normalization to endogenous counts
# ─────────────────────────────────────────────────────────────────────────────
 
print("\n" + "=" * 60)
print("APPLYING NORMALIZATION TO ENDOGENOUS COUNTS")
print("=" * 60)
 
# Multiply each sample's counts by its normalization factor
# df_endo: rows = genes, columns = samples
df_endo_technorm = df_endo.multiply(norm_factors, axis=1)
 
print(f"\n  Raw counts range       : {df_endo.values.min():.1f} – {df_endo.values.max():.1f}")
print(f"  Normalized counts range: {df_endo_technorm.values.min():.2f} – {df_endo_technorm.values.max():.2f}")
print(f"\n  Shape check: {df_endo_technorm.shape[0]} genes × {df_endo_technorm.shape[1]} samples")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Save outputs
# ─────────────────────────────────────────────────────────────────────────────
 
r2_series.to_csv("pos_ctrl_linearity.csv",               header=True)
norm_factors.to_csv("pos_ctrl_normalization_factors.csv", header=True)
df_endo_technorm.to_csv("counts_endogenous_technorm.csv")
 
print("\n" + "=" * 60)
print("OUTPUT FILES SAVED")
print("=" * 60)
print("  pos_ctrl_linearity.csv")
print("  pos_ctrl_normalization_factors.csv")
print("  counts_endogenous_technorm.csv")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. QC plot
# ─────────────────────────────────────────────────────────────────────────────
 
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Step 2: Positive Control QC", fontsize=13, fontweight="bold")
 
# (a) R² per sample
samples      = r2_series.index.tolist()
r2_values    = r2_series.values
bar_colors   = ["#e74c3c" if v < R2_THRESHOLD else "#2ecc71" for v in r2_values]
 
axes[0].bar(range(len(samples)), r2_values, color=bar_colors, alpha=0.85, width=0.8)
axes[0].axhline(R2_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
                label=f"Threshold (R² = {R2_THRESHOLD})")
axes[0].set_title("Positive Control Linearity (R²) per Sample")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("R²")
axes[0].set_ylim(0.8, 1.01)
axes[0].legend(fontsize=9)
axes[0].tick_params(axis='x', labelbottom=False)
 
# (b) Normalization factors per sample
nf_values  = norm_factors.values
bar_colors2 = ["#3498db"] * len(nf_values)
 
axes[1].bar(range(len(nf_values)), nf_values, color=bar_colors2, alpha=0.85, width=0.8)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Factor = 1.0 (no correction)")
axes[1].set_title("Positive Control Normalization Factors per Sample")
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Normalization factor")
axes[1].legend(fontsize=9)
axes[1].tick_params(axis='x', labelbottom=False)
 
plt.tight_layout()
plt.savefig("step2_positive_control_QC.png", dpi=150, bbox_inches="tight")
plt.close()
 
print("  step2_positive_control_QC.png")
print()
print("Next step: run step3_background_threshold.py")

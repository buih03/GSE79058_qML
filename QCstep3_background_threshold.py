"""
Step 3: Background thresholding
================================
Uses the 8 NanoString negative control probes to:
  1. Compute a per-sample background noise threshold (mean + 2*SD of negative controls)
  2. Floor any endogenous gene count below that threshold to the threshold value
  3. Apply the same flooring logic to the negative controls themselves (sanity check only)

This step operates on the TECHNICALLY NORMALIZED counts from Step 2
(counts_endogenous_technorm.csv), since background thresholding should
happen after correcting for lane-to-lane variation, not before.

Inputs:
  counts_negative.csv               : 8 negative control probes × 76 samples (RAW)
  counts_endogenous_technorm.csv    : 730 genes × 76 samples (after Step 2 normalization)
  pos_ctrl_normalization_factors.csv: normalization factors from Step 2 (to also normalize negatives)

Outputs:
  background_thresholds.csv             : per-sample background threshold
  counts_endogenous_bgfloored.csv       : endogenous counts after flooring
  step3_background_QC.png               : visual summary of flooring impact

Usage:
  python step3_background_threshold.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load inputs
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3: Background Thresholding")
print("=" * 60)

df_neg_raw   = pd.read_csv("counts_negative.csv",                index_col=0)
df_endo_tn   = pd.read_csv("counts_endogenous_technorm.csv",     index_col=0)
norm_factors = pd.read_csv("pos_ctrl_normalization_factors.csv", index_col=0).iloc[:, 0]

print(f"\n  Negative controls loaded (raw)     : {df_neg_raw.shape[0]} probes × {df_neg_raw.shape[1]} samples")
print(f"  Endogenous counts loaded (tech-norm): {df_endo_tn.shape[0]} genes  × {df_endo_tn.shape[1]} samples")

# Apply the SAME Step 2 normalization factors to the negative controls,
# so the background threshold is computed on the same scale as the
# endogenous counts we're about to floor.
df_neg_tn = df_neg_raw.multiply(norm_factors, axis=1)
print(f"\n  Negative controls scaled using Step 2 normalization factors (for consistent units)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Compute per-sample background threshold
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("COMPUTING BACKGROUND THRESHOLDS (mean + 2*SD of negative controls)")
print("=" * 60)

neg_mean = df_neg_tn.mean(axis=0)   # mean across 8 probes, per sample
neg_std  = df_neg_tn.std(axis=0)    # SD across 8 probes, per sample

threshold = neg_mean + 2 * neg_std
threshold.name = "background_threshold"

print(f"\n  Threshold range : {threshold.min():.2f} – {threshold.max():.2f}")
print(f"  Mean threshold  : {threshold.mean():.2f}")
print(f"\n  Sample thresholds (first 10):")
print(threshold.head(10).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Apply flooring to endogenous counts
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("APPLYING BACKGROUND FLOOR TO ENDOGENOUS COUNTS")
print("=" * 60)

# For each sample (column), floor any gene count below that sample's threshold
df_endo_floored = df_endo_tn.copy()

n_floored_per_sample = {}
for sample in df_endo_floored.columns:
    thresh = threshold[sample]
    col = df_endo_floored[sample]
    below_mask = col < thresh
    n_floored_per_sample[sample] = int(below_mask.sum())
    df_endo_floored.loc[below_mask, sample] = thresh

total_floored = sum(n_floored_per_sample.values())
total_values  = df_endo_floored.shape[0] * df_endo_floored.shape[1]
pct_floored   = 100 * total_floored / total_values

print(f"\n  Total gene-sample values floored : {total_floored:,} / {total_values:,} ({pct_floored:.2f}%)")
print(f"  Average genes floored per sample : {np.mean(list(n_floored_per_sample.values())):.1f} (of 730)")

# Show a few samples with the most flooring (potential low-quality samples)
floored_series = pd.Series(n_floored_per_sample).sort_values(ascending=False)
print(f"\n  Samples with the most genes floored (top 5):")
print(floored_series.head(5).to_string())

print(f"\n  Samples with the least genes floored (top 5):")
print(floored_series.tail(5).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save outputs
# ─────────────────────────────────────────────────────────────────────────────

threshold.to_csv("background_thresholds.csv", header=True)
df_endo_floored.to_csv("counts_endogenous_bgfloored.csv")

print("\n" + "=" * 60)
print("OUTPUT FILES SAVED")
print("=" * 60)
print("  background_thresholds.csv")
print("  counts_endogenous_bgfloored.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 5. QC plot
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Step 3: Background Thresholding QC", fontsize=13, fontweight="bold")

# (a) Background threshold per sample
axes[0].bar(range(len(threshold)), threshold.values, color="#9b59b6", alpha=0.85, width=0.8)
axes[0].set_title("Background Threshold per Sample")
axes[0].set_xlabel("Sample index")
axes[0].set_ylabel("Threshold (normalized counts)")
axes[0].tick_params(axis='x', labelbottom=False)

# (b) Number of genes floored per sample
axes[1].bar(range(len(floored_series)), floored_series.values, color="#e67e22", alpha=0.85, width=0.8)
axes[1].set_title("Number of Genes Floored per Sample (out of 730)")
axes[1].set_xlabel("Sample (sorted by # floored)")
axes[1].set_ylabel("Genes floored")

plt.tight_layout()
plt.savefig("step3_background_QC.png", dpi=150, bbox_inches="tight")
plt.close()

print("  step3_background_QC.png")
print()
print("Next step: run step4_housekeeping_norm.py")

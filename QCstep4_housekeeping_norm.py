"""
Step 4: Housekeeping Gene Normalization
=========================================
Uses the 40 NanoString housekeeping genes to:
  1. Check which housekeeping genes are stable across control/ILC1/ILC2
     (using one-way ANOVA per gene)
  2. Compute a per-sample normalization factor from the STABLE subset only
  3. Apply that factor to the background-floored endogenous counts from Step 3

Inputs:
  counts_housekeeping.csv             : 40 housekeeping genes × 76 samples (RAW)
  counts_endogenous_bgfloored.csv     : 730 genes × 76 samples (after Step 3)
  pos_ctrl_normalization_factors.csv  : Step 2 normalization factors (for consistent units)

  Sample-to-group mapping is embedded directly in this script (see SAMPLE_GROUPS
  below), reconstructed from the GEO2R "Tumor grade" column:
      grade 1  -> ILC1
      grade 2  -> ILC2
      NA       -> control

Outputs:
  metadata.csv                          : sample_id, patient, group (for reuse in later steps)
  housekeeping_anova_results.csv        : F-stat and p-value per housekeeping gene
  housekeeping_stable_genes.txt         : list of genes used for normalization
  hk_normalization_factors.csv          : per-sample normalization factor
  counts_endogenous_final.csv           : FULLY normalized endogenous counts (ready for ML)
  step4_housekeeping_QC.png             : QC plots

Usage:
  python step4_housekeeping_norm.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


# ─────────────────────────────────────────────────────────────────────────────
# 0. Sample -> group mapping (reconstructed from GEO2R "Tumor grade" column)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_GROUPS = {
    "GSM2084669": "ILC1", "GSM2084670": "ILC1", "GSM2084671": "ILC1",
    "GSM2084672": "ILC1", "GSM2084673": "ILC1", "GSM2084674": "ILC1",
    "GSM2084675": "ILC1", "GSM2084676": "ILC1", "GSM2084677": "ILC1",
    "GSM2084678": "control",
    "GSM2084679": "ILC2", "GSM2084680": "ILC2", "GSM2084681": "ILC2",
    "GSM2084682": "ILC1", "GSM2084683": "ILC1", "GSM2084684": "ILC1",
    "GSM2084685": "control",
    "GSM2084686": "ILC2", "GSM2084687": "ILC2", "GSM2084688": "ILC2",
    "GSM2084689": "ILC1", "GSM2084690": "ILC1", "GSM2084691": "ILC1",
    "GSM2084692": "ILC2", "GSM2084693": "ILC2", "GSM2084694": "ILC2",
    "GSM2084695": "ILC1", "GSM2084696": "ILC1", "GSM2084697": "ILC1",
    "GSM2084698": "control",
    "GSM2084699": "ILC2", "GSM2084700": "ILC2", "GSM2084701": "ILC2",
    "GSM2084702": "ILC2", "GSM2084703": "ILC2", "GSM2084704": "ILC2",
    "GSM2084705": "ILC1", "GSM2084706": "ILC1", "GSM2084707": "ILC1",
    "GSM2084708": "ILC1", "GSM2084709": "ILC1", "GSM2084710": "ILC1",
    "GSM2084711": "control",
    "GSM2084712": "ILC1", "GSM2084713": "ILC1", "GSM2084714": "ILC1",
    "GSM2084715": "ILC1", "GSM2084716": "ILC1", "GSM2084717": "ILC1",
    "GSM2084718": "control",
    "GSM2084719": "ILC1", "GSM2084720": "ILC1", "GSM2084721": "ILC1",
    "GSM2084722": "ILC1", "GSM2084723": "ILC1", "GSM2084724": "ILC1",
    "GSM2084725": "ILC2", "GSM2084726": "ILC2", "GSM2084727": "ILC2",
    "GSM2084728": "ILC1", "GSM2084729": "ILC1", "GSM2084730": "ILC1",
    "GSM2084731": "ILC2", "GSM2084732": "ILC2", "GSM2084733": "ILC2",
    "GSM2084734": "ILC2", "GSM2084735": "ILC2", "GSM2084736": "ILC2",
    "GSM2084737": "control",
    "GSM2084738": "ILC2", "GSM2084739": "ILC2", "GSM2084740": "ILC2",
    "GSM2084741": "ILC2", "GSM2084742": "ILC2", "GSM2084743": "ILC2",
    "GSM2084744": "control",
}

ANOVA_P_THRESHOLD = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load inputs
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4: Housekeeping Gene Normalization")
print("=" * 60)

df_hk_raw    = pd.read_csv("counts_housekeeping.csv",         index_col=0)
df_endo_bg   = pd.read_csv("counts_endogenous_bgfloored.csv", index_col=0)
norm_factors = pd.read_csv("pos_ctrl_normalization_factors.csv", index_col=0).iloc[:, 0]

print(f"\n  Housekeeping genes loaded (raw)        : {df_hk_raw.shape[0]} genes × {df_hk_raw.shape[1]} samples")
print(f"  Endogenous counts loaded (bg-floored)  : {df_endo_bg.shape[0]} genes × {df_endo_bg.shape[1]} samples")

# Apply Step 2 normalization factors to housekeeping genes for consistent units
df_hk_tn = df_hk_raw.multiply(norm_factors, axis=1)
print(f"  Housekeeping counts scaled using Step 2 normalization factors")

# Build metadata table and save it for reuse in later steps (ML scripts etc.)
metadata = pd.DataFrame({
    "sample_id": list(SAMPLE_GROUPS.keys()),
    "group":     list(SAMPLE_GROUPS.values()),
})
metadata["patient"] = metadata["sample_id"].map(
    lambda s: None  # placeholder; patient ID isn't strictly needed downstream here
)
metadata = metadata.drop(columns=["patient"])
metadata.to_csv("metadata.csv", index=False)
print(f"\n  metadata.csv saved ({len(metadata)} samples)")
print(f"  Group counts: {metadata['group'].value_counts().to_dict()}")

# Align group labels to the sample order in df_hk_tn
groups = pd.Series(SAMPLE_GROUPS).reindex(df_hk_tn.columns)
assert not groups.isnull().any(), "Some samples are missing a group label!"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANOVA stability check per housekeeping gene
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CHECKING HOUSEKEEPING GENE STABILITY (one-way ANOVA across groups)")
print("=" * 60)

anova_results = []

for gene in df_hk_tn.index:
    values = df_hk_tn.loc[gene]
    log_values = np.log2(values + 1)

    group_arrays = [log_values[groups == g].values for g in ["control", "ILC1", "ILC2"]]
    f_stat, p_val = f_oneway(*group_arrays)

    anova_results.append({"gene": gene, "F_stat": f_stat, "p_value": p_val})

df_anova = pd.DataFrame(anova_results).set_index("gene").sort_values("p_value")

n_unstable = (df_anova["p_value"] < ANOVA_P_THRESHOLD).sum()
n_stable   = len(df_anova) - n_unstable

print(f"\n  ANOVA p-value threshold : {ANOVA_P_THRESHOLD}")
print(f"  Stable genes (p ≥ {ANOVA_P_THRESHOLD})   : {n_stable} / {len(df_anova)}")
print(f"  Unstable genes (p < {ANOVA_P_THRESHOLD})  : {n_unstable} / {len(df_anova)}")

if n_unstable > 0:
    print(f"\n  Unstable housekeeping genes (excluded from normalization):")
    print(df_anova[df_anova["p_value"] < ANOVA_P_THRESHOLD].to_string())

stable_genes = df_anova[df_anova["p_value"] >= ANOVA_P_THRESHOLD].index.tolist()

if len(stable_genes) < 5:
    print(f"\n  WARNING: Only {len(stable_genes)} stable housekeeping genes remain. "
          f"This is low — consider relaxing the threshold or reviewing manually.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Compute normalization factors from stable genes only
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("COMPUTING NORMALIZATION FACTORS (stable housekeeping genes only)")
print("=" * 60)

df_hk_stable = df_hk_tn.loc[stable_genes]

def geom_mean(values):
    vals = np.array(values, dtype=float)
    vals = np.where(vals <= 0, np.nan, vals)
    return np.exp(np.nanmean(np.log(vals)))

sample_geom_means = df_hk_stable.apply(lambda col: geom_mean(col.values), axis=0)
reference_geom_mean = geom_mean(sample_geom_means.values)

hk_norm_factors = reference_geom_mean / sample_geom_means
hk_norm_factors.name = "hk_norm_factor"

print(f"\n  Reference geometric mean: {reference_geom_mean:.2f}")
print(f"  Normalization factor range: {hk_norm_factors.min():.4f} – {hk_norm_factors.max():.4f}")

# Check patient 8 samples specifically (flagged in Step 3)
patient8_samples = ["GSM2084719", "GSM2084720", "GSM2084721", "GSM2084722", "GSM2084723", "GSM2084724"]
patient8_factors = hk_norm_factors[hk_norm_factors.index.isin(patient8_samples)]
print(f"\n  Patient 8 housekeeping normalization factors (flagged in Step 3):")
print(patient8_factors.to_string())
print(f"  (Factor > 1 means low housekeeping signal -> being scaled UP -> "
      f"consistent with a technical RNA-input issue if values are elevated here)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Apply normalization to background-floored endogenous counts
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("APPLYING HOUSEKEEPING NORMALIZATION TO ENDOGENOUS COUNTS")
print("=" * 60)

df_endo_final = df_endo_bg.multiply(hk_norm_factors, axis=1)

print(f"\n  Pre-normalization range : {df_endo_bg.values.min():.2f} – {df_endo_bg.values.max():.2f}")
print(f"  Final normalized range  : {df_endo_final.values.min():.2f} – {df_endo_final.values.max():.2f}")
print(f"  Shape check: {df_endo_final.shape[0]} genes × {df_endo_final.shape[1]} samples")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save outputs
# ─────────────────────────────────────────────────────────────────────────────

df_anova.to_csv("housekeeping_anova_results.csv")
with open("housekeeping_stable_genes.txt", "w") as f:
    f.write("\n".join(stable_genes))
hk_norm_factors.to_csv("hk_normalization_factors.csv", header=True)
df_endo_final.to_csv("counts_endogenous_final.csv")

print("\n" + "=" * 60)
print("OUTPUT FILES SAVED")
print("=" * 60)
print("  metadata.csv")
print("  housekeeping_anova_results.csv")
print("  housekeeping_stable_genes.txt")
print("  hk_normalization_factors.csv")
print("  counts_endogenous_final.csv   <-- ready for ML scripts")


# ─────────────────────────────────────────────────────────────────────────────
# 6. QC plots
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Step 4: Housekeeping Gene Normalization QC", fontsize=13, fontweight="bold")

# (a) ANOVA p-values per gene, sorted
sorted_p = df_anova["p_value"].sort_values()
bar_colors = ["#e74c3c" if p < ANOVA_P_THRESHOLD else "#2ecc71" for p in sorted_p.values]
axes[0].bar(range(len(sorted_p)), sorted_p.values, color=bar_colors, alpha=0.85, width=0.8)
axes[0].axhline(ANOVA_P_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
                label=f"Threshold (p = {ANOVA_P_THRESHOLD})")
axes[0].set_title("Housekeeping Gene ANOVA p-values\n(red = unstable, excluded)")
axes[0].set_xlabel("Housekeeping gene (sorted)")
axes[0].set_ylabel("p-value")
axes[0].legend(fontsize=9)

# (b) HK normalization factors per sample
axes[1].bar(range(len(hk_norm_factors)), hk_norm_factors.values, color="#16a085", alpha=0.85, width=0.8)
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Factor = 1.0")
axes[1].set_title("Housekeeping Normalization Factor per Sample")
axes[1].set_xlabel("Sample index")
axes[1].set_ylabel("Normalization factor")
axes[1].legend(fontsize=9)
axes[1].tick_params(axis='x', labelbottom=False)

plt.tight_layout()
plt.savefig("step4_housekeeping_QC.png", dpi=150, bbox_inches="tight")
plt.close()

print("  step4_housekeeping_QC.png")
print()
print("=" * 60)
print("NORMALIZATION PIPELINE COMPLETE")
print("=" * 60)
print("counts_endogenous_final.csv contains fully normalized data:")
print("  Technical (positive control) normalization")
print("  Background thresholding")
print("  Housekeeping (biological/RNA input) normalization")
print()
print("Remaining decision before ML: how to handle the 3 cores (a/b/c) per tumor")
print("focus — average them, or stratify train/test splits by patient.")

"""
Step 1: Parse NanoString RCC.gz Files
======================================
Read all 76 .RCC.gz files in the current folder and extracts:
  - Lane QC metrics (FOV rate, binding density)
  - Positive control counts (for technical normalization)
  - Negative control counts (for background thresholding)
  - Housekeeping gene counts (for biological normalization)
  - Endogenous gene counts (the 730 genes of interest)

Outputs (saved to current folder):
  - counts_endogenous.csv     : 730 genes × 76 samples
  - counts_positive.csv       : positive controls × 76 samples
  - counts_negative.csv       : negative controls × 76 samples
  - counts_housekeeping.csv   : housekeeping genes × 76 samples
  - lane_qc.csv               : 76 samples × QC metrics
 
Usage:
  python step1_parse_rcc.py
"""

import gzip
import glob
import os
import pandas as pd

def parse_rcc(filepath):
    """
    Parse one .RCC.gz file and return its contents as a dictionary.
 
    Returns:
        dict with keys:
            'sample_id'   : GSM accession extracted from filename
            'lane'        : dict of lane QC metrics
            'counts'      : dict of {CodeClass: {GeneName: count}}
    """
    sample_id = os.path.basename(filepath).split("_")[0]  # e.g. GSM2084669
 
    lane = {}
    counts = {}   # will hold: {'Endogenous': {}, 'Positive': {}, 'Negative': {}, 'Housekeeping': {}}

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        current_section = None
 
        for line in f:
            line = line.strip()
 
            # Detect section boundaries
            if line.startswith("<") and not line.startswith("</"):
                current_section = line.strip("<>")
                continue
            if line.startswith("</"):
                current_section = None
                continue
 
            # Parse Lane_Attributes section
            if current_section == "Lane_Attributes":
                if "," in line:
                    key, value = line.split(",", 1)
                    lane[key.strip()] = value.strip()
 
            # Parse Code_Summary section
            if current_section == "Code_Summary":
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                code_class = parts[0].strip()
                gene_name  = parts[1].strip()
                # parts[2] is accession — we don't need it
                try:
                    count = float(parts[3].strip())
                except ValueError:
                    continue  # skip header row
 
                if code_class not in counts:
                    counts[code_class] = {}
                counts[code_class][gene_name] = count
 
    return {
        "sample_id": sample_id,
        "lane":      lane,
        "counts":    counts
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
 
    # Find all RCC.gz files in the current folder
    rcc_files = sorted(glob.glob("*.RCC.gz"))
 
    if len(rcc_files) == 0:
        raise FileNotFoundError(
            "No .RCC.gz files found in the current folder. "
            "Make sure you are running this script from the folder "
            "containing your RCC.gz files."
        )
 
    print(f"Found {len(rcc_files)} RCC.gz files. Parsing...")
 
    # Storage
    lane_records       = []
    endo_records       = {}   # {sample_id: {gene: count}}
    pos_records        = {}
    neg_records        = {}
    hk_records         = {}
 
    for filepath in rcc_files:
        result    = parse_rcc(filepath)
        sid       = result["sample_id"]
        lane      = result["lane"]
        counts    = result["counts"]
 
        print(f"  Parsed {sid}")
 
        # Lane QC metrics
        lane_records.append({
            "sample_id":       sid,
            "FovCount":        float(lane.get("FovCount",    "nan")),
            "FovCounted":      float(lane.get("FovCounted",  "nan")),
            "BindingDensity":  float(lane.get("BindingDensity", "nan")),
        })
 
        # Count tables by CodeClass
        endo_records[sid] = counts.get("Endogenous",   {})
        pos_records[sid]  = counts.get("Positive",     {})
        neg_records[sid]  = counts.get("Negative",     {})
        hk_records[sid]   = counts.get("Housekeeping", {})
 
    print("\nAll files parsed successfully.\n")
 
    # ── Build DataFrames ──────────────────────────────────────────────────────
    # Convention: rows = genes/probes, columns = samples
 
    df_endo = pd.DataFrame(endo_records).fillna(0).astype(float)
    df_pos  = pd.DataFrame(pos_records).fillna(0).astype(float)
    df_neg  = pd.DataFrame(neg_records).fillna(0).astype(float)
    df_hk   = pd.DataFrame(hk_records).fillna(0).astype(float)
    df_lane = pd.DataFrame(lane_records).set_index("sample_id")
 
    # Add FOV rate column (QC ratio)
    df_lane["FovRate"] = df_lane["FovCounted"] / df_lane["FovCount"]
 
    # ── Print summary ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("PARSING SUMMARY")
    print("=" * 55)
    print(f"  Endogenous genes   : {df_endo.shape[0]} genes × {df_endo.shape[1]} samples")
    print(f"  Positive controls  : {df_pos.shape[0]} probes × {df_pos.shape[1]} samples")
    print(f"  Negative controls  : {df_neg.shape[0]} probes × {df_neg.shape[1]} samples")
    print(f"  Housekeeping genes : {df_hk.shape[0]} genes  × {df_hk.shape[1]} samples")
    print()
 
    print("  Positive control probe names:")
    for name in df_pos.index:
        print(f"    {name}")
    print()
 
    print("  Negative control probe names:")
    for name in df_neg.index:
        print(f"    {name}")
    print()
 
    print("  Housekeeping gene names:")
    for name in df_hk.index:
        print(f"    {name}")
    print()
 
    # ── Lane QC preview ───────────────────────────────────────────────────────
    print("=" * 55)
    print("LANE QC METRICS (first 10 samples)")
    print("=" * 55)
    print(df_lane[["FovCount", "FovCounted", "FovRate", "BindingDensity"]].head(10).to_string())
    print()
 
    # Flag samples outside acceptable thresholds
    fov_fail     = df_lane[df_lane["FovRate"] < 0.75]
    density_fail = df_lane[
        (df_lane["BindingDensity"] < 0.1) |
        (df_lane["BindingDensity"] > 2.25)
    ]
 
    if len(fov_fail) > 0:
        print(f"  {len(fov_fail)} sample(s) with FOV rate < 0.75:")
        print(fov_fail[["FovRate"]].to_string())
    else:
        print("  All samples pass FOV rate threshold (≥ 0.75)")
 
    if len(density_fail) > 0:
        print(f"  {len(density_fail)} sample(s) with binding density outside 0.1–2.25:")
        print(density_fail[["BindingDensity"]].to_string())
    else:
        print("  All samples pass binding density threshold (0.1–2.25)")
 
    print()
 
    # ── Save outputs ──────────────────────────────────────────────────────────
    df_endo.to_csv("counts_endogenous.csv")
    df_pos.to_csv("counts_positive.csv")
    df_neg.to_csv("counts_negative.csv")
    df_hk.to_csv("counts_housekeeping.csv")
    df_lane.to_csv("lane_qc.csv")
 
    print("=" * 55)
    print("OUTPUT FILES SAVED")
    print("=" * 55)
    print("  counts_endogenous.csv")
    print("  counts_positive.csv")
    print("  counts_negative.csv")
    print("  counts_housekeeping.csv")
    print("  lane_qc.csv")
    print()
    print("Next step: run step2_lane_qc.py")
 
 
if __name__ == "__main__":
    main()
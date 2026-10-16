"""
Unsupervised heuristics for choosing the number of PCs on the 730-gene ILC
data, plus a seed-averaged ARI check of the candidates.
 
Usage:
  python pca_component_heuristics.py --data-dir .
"""
 
import argparse
from pathlib import Path
 
import numpy as np
import pandas as pd
 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
 
def load_ilc730_data(data_dir):
    data_dir = Path(data_dir)
    expr = pd.read_csv(data_dir / "counts_endogenous_final.csv", index_col=0)
    meta = pd.read_csv(data_dir / "metadata.csv")
 
    X = expr.T.loc[meta["sample_id"]]
    y = meta["group"].values
 
    X_scaled = StandardScaler().fit_transform(np.log1p(X.values))
    return X_scaled, y
 
 
def eig_fractions(X):
    # eigenvalues of rho = C / trace(C), same quantity qPCA phase-estimates
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    return lam / lam.sum()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Heuristics (each returns a number of PCs to keep)
# ─────────────────────────────────────────────────────────────────────────────
 
def h_variance(frac, target):
    return int(np.searchsorted(np.cumsum(frac), target) + 1)
 
 
def h_kaiser(frac, n_genes):
    # eigenvalue > 1 on the correlation matrix == fraction > 1/p
    return int((frac > 1.0 / n_genes).sum())
 
 
def h_broken_stick(frac):
    r = len(frac)
    stick = np.array([np.sum(1.0 / np.arange(k, r + 1)) / r for k in range(1, r + 1)])
    above = frac > stick
    return int(np.argmin(above)) if not above.all() else r
 
 
def h_elbow(frac, max_k=50):
    y = frac[:max_k]
    xn = np.arange(max_k) / (max_k - 1)
    yn = (y - y.min()) / (y.max() - y.min())
    return int(np.argmax((1 - xn) - yn) + 1)  # farthest point below the chord
 
 
def h_parallel(X, frac, n_perm, rng, q=95):
    r = len(frac)
    null = np.empty((n_perm, r))
    for i in range(n_perm):
        null[i] = eig_fractions(rng.permuted(X, axis=0))[:r]
    above = frac > np.percentile(null, q, axis=0)
    return int(np.argmin(above)) if not above.all() else r
 
 
def h_qpe_resolution(frac, n_counting):
    # eigenvalues below one phase bin (2^-m) can't be resolved by QPE
    return int((frac >= 2.0 ** -n_counting).sum())
 
 
def h_separation(frac, n_counting, bins=2):
    # keep PCs while each one is >= `bins` phase bins above its next neighbour
    ok = (frac[:-1] - frac[1:]) >= bins * 2.0 ** -n_counting
    return int(np.argmin(ok)) if not ok.all() else len(ok)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ARI sanity check (labels used for reporting only, not for selection)
# ─────────────────────────────────────────────────────────────────────────────
 
def ari_over_seeds(scores, y, n_seeds):
    true_numeric = pd.factorize(y)[0]
    k_clusters = len(np.unique(y))
    aris = [
        adjusted_rand_score(
            true_numeric,
            KMeans(n_clusters=k_clusters, random_state=s, n_init=10).fit_predict(scores),
        )
        for s in range(n_seeds)
    ]
    return np.mean(aris), np.std(aris)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--n-counting", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()
 
    X, y = load_ilc730_data(args.data_dir)
    n, p = X.shape
    frac = eig_fractions(X)[: n - 1]  # last eigenvalue is ~0 after centering
    rng = np.random.default_rng(0)
 
    picks = {
        "Cumulative variance >= 80%": h_variance(frac, 0.80),
        "Cumulative variance >= 90%": h_variance(frac, 0.90),
        "Kaiser (eigenvalue > 1)": h_kaiser(frac, p),
        "Broken stick": h_broken_stick(frac),
        "Scree elbow (first 50)": h_elbow(frac),
        "Parallel analysis (95th pct)": h_parallel(X, frac, args.n_perm, rng),
        f"qPE resolution (>= 2^-{args.n_counting})": h_qpe_resolution(frac, args.n_counting),
        "qPE separation (gap >= 2 bins)": h_separation(frac, args.n_counting),
    }
 
    cum = np.cumsum(frac)
    print(f"{'heuristic':<34} {'k':>4} {'cum. var.':>10}")
    for name, k in picks.items():
        print(f"{name:<34} {k:>4} {(cum[k-1] if k > 0 else 0.0):>9.1%}")
 
    bin_w = 2.0 ** -args.n_counting
    gaps = frac[:-1] - frac[1:]
    print(f"\nEigenvalue gaps (1 phase bin = {bin_w:.4f})")
    print(f"{'PC':>4} {'eigenvalue':>11} {'gap to next':>12} {'gap (bins)':>11}")
    for i in range(min(15, len(gaps))):
        print(f"{i+1:>4} {frac[i]:>11.4f} {gaps[i]:>12.4f} {gaps[i] / bin_w:>11.2f}")
 
    ks = sorted(set([2, 3, 5, 10] + [k for k in picks.values() if 0 < k <= 75]))
 
    print(f"\n{'k':>4} {'min gap':>9} {'counting qubits for 2-bin gaps':>32}")
    for k in [k for k in ks if k < len(frac)]:
        g = gaps[:k].min()
        print(f"{k:>4} {g:>9.4f} {int(np.ceil(np.log2(2.0 / g))):>32}")
 
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    scores_full = U * s
 
    print(f"\nARI over {args.n_seeds} KMeans seeds (reporting only)")
    print(f"{'k':>4} {'cum. var.':>10} {'ARI mean':>10} {'ARI sd':>8}")
    for k in ks:
        m, sd = ari_over_seeds(scores_full[:, :k], y, args.n_seeds)
        print(f"{k:>4} {cum[k-1]:>9.1%} {m:>10.4f} {sd:>8.4f}")
 
 
if __name__ == "__main__":
    main()
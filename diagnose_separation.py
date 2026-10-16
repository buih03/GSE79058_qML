"""
Why does classical PCA look separated in one analysis and not in another?
Compares preprocessing variants (including a DESeq2-style one), with an optional list of excluded samples.

Usage:
  python diagnose_separation.py --data-dir .
  python diagnose_separation.py --data-dir . --exclude excluded_samples.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

COLORS = {"control": "#F8766D", "ILC1": "#00BA38", "ILC2": "#619CFF"}


def load(data_dir, exclude):
    expr = pd.read_csv(Path(data_dir) / "counts_endogenous_final.csv", index_col=0)  # genes x samples
    meta = pd.read_csv(Path(data_dir) / "metadata.csv")
    if exclude:
        if not Path(exclude).exists():
            raise SystemExit(f"{exclude} not found. Create it with one sample_id per line, or omit --exclude.")
        drop = set(Path(exclude).read_text().split())
        meta = meta[~meta["sample_id"].isin(drop)]
    X = expr.T.loc[meta["sample_id"]].values.astype(float)
    return X, meta["group"].values


def size_factors(X):
    # median-of-ratios, as in DESeq2
    with np.errstate(divide="ignore"):
        logX = np.log(X)
    ref = logX.mean(0)
    ok = np.isfinite(ref)
    return np.exp(np.median(logX[:, ok] - ref[ok], axis=1))


def deseq_style(X, ntop=None):
    Z = np.log2(X / size_factors(X)[:, None] + 1)
    if ntop:
        Z = Z[:, np.argsort(Z.var(0))[::-1][:ntop]]
    return Z


def variants(X):
    cpm = X / X.sum(1, keepdims=True) * 1e6
    return {
        "log1p + z-score (pipeline)": StandardScaler().fit_transform(np.log1p(X)),
        "log1p, no scaling": np.log1p(X),
        "CPM, log2(x+1), z-score": StandardScaler().fit_transform(np.log2(cpm + 1)),
        "DESeq2-style, top 500 genes, no scaling": deseq_style(X, 500),
        "DESeq2-style, all genes, no scaling": deseq_style(X),
    }


def metrics(Z, y, X):
    pca = PCA(n_components=2)
    S = pca.fit_transform(Z)
    two = np.where(y == "control", "control", "ILC")
    km3 = KMeans(3, n_init=10, random_state=0).fit_predict(S)
    km2 = KMeans(2, n_init=10, random_state=0).fit_predict(S)
    logtot = np.log(X.sum(1))
    return S, {
        "PC1 var": pca.explained_variance_ratio_[0],
        "PC2 var": pca.explained_variance_ratio_[1],
        "silhouette, 3 groups": silhouette_score(S, y),
        "silhouette, control vs ILC": silhouette_score(S, two),
        "ARI, 3 groups (KMeans k=3)": adjusted_rand_score(y, km3),
        "ARI, control vs ILC (KMeans k=2)": adjusted_rand_score(two, km2),
        "|corr| PC1 vs log total counts": abs(np.corrcoef(S[:, 0], logtot)[0, 1]),
        "|corr| PC2 vs log total counts": abs(np.corrcoef(S[:, 1], logtot)[0, 1]),
    }


def ellipse(ax, pts, color):
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    size = 2 * np.sqrt(5.991 * w[::-1])  # 95% data ellipse
    ax.add_patch(Ellipse(pts.mean(0), size[0], size[1], angle=ang, fill=True,
                         facecolor=color, alpha=0.12, edgecolor=color, lw=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--exclude", default=None, help="text file, one sample_id per line")
    ap.add_argument("--out", default="fig_separation_diagnostic.png")
    args = ap.parse_args()

    X, y = load(args.data_dir, args.exclude)
    print("samples:", {g: int((y == g).sum()) for g in COLORS}, "total", len(y))

    rows, fig_data = {}, {}
    for name, Z in variants(X).items():
        S, m = metrics(Z, y, X)
        rows[name], fig_data[name] = m, S
    table = pd.DataFrame(rows).round(3)
    print(table.to_string())
    table.to_csv(Path(args.out).with_suffix(".csv"))

    ncols = 3
    nrows = int(np.ceil(len(fig_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows), squeeze=False)
    for ax in axes.ravel()[len(fig_data):]:
        ax.axis("off")
    for ax, (name, S) in zip(axes.ravel(), fig_data.items()):
        for g, c in COLORS.items():
            pts = S[y == g]
            ax.scatter(pts[:, 0], pts[:, 1], s=26, color=c, alpha=0.85, label=g,
                       edgecolors="white", linewidths=0.4)
            ellipse(ax, pts, c)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel(f"PC1 ({rows[name]['PC1 var']:.0%})")
        ax.set_ylabel(f"PC2 ({rows[name]['PC2 var']:.0%})")
        ax.autoscale()
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
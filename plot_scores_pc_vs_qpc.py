"""
2D views of classical PCA versus qPCA for each (k, m) setting.
  1. Sample scores: classical PCs, qPCs, and an overlay (default PC1 vs PC2).
  2. Gene loadings: classical vs quantum weights per component, with y = x line.

Usage:
  python plot_scores_pc_vs_qpc.py --data-dir . --module qpca_ilc730_v2
  python plot_scores_pc_vs_qpc.py --data-dir . --module qpca_ilc730_v2 --pair 3 4
"""

import argparse
import importlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"control": ("#0072B2", "s"), "ILC1": ("#E69F00", "o"), "ILC2": ("#009E73", "^")}


def run_configs(P, data_dir, configs):
    X_padded, X_scaled, y, _ = P.load_ilc730_data(data_dir)
    classical = P.classical_pca(X_scaled, y, max(k for k, _ in configs))
    out = {}
    for k, m in configs:
        print(f"k={k}, m={m}", flush=True)
        q = P.quantum_pca_statevector(X_padded, X_scaled, y, k, m)
        Wc = classical["components"][:k]
        Wq = q["components_unpadded"].copy()
        sign = np.sign(np.sum(Wc * Wq, axis=1))  # global sign of an eigenvector is arbitrary
        sign[sign == 0] = 1
        Wq *= sign[:, None]
        out[(k, m)] = {"Wc": Wc, "Wq": Wq, "Sc": X_scaled @ Wc.T, "Sq": X_scaled @ Wq.T}
    return out, y, classical["explained_variance_ratio"]


def scatter_groups(ax, xy, y, hollow=False, size=34):
    for g, (color, marker) in STYLE.items():
        idx = y == g
        if hollow:
            ax.scatter(xy[idx, 0], xy[idx, 1], s=size, marker=marker, facecolors="none",
                       edgecolors=color, linewidths=1.2)
        else:
            ax.scatter(xy[idx, 0], xy[idx, 1], s=size, marker=marker, color=color,
                       alpha=0.85, edgecolors="white", linewidths=0.4, label=g)


def plot_scores(res, y, evr, pair, out):
    a, b = pair[0] - 1, pair[1] - 1
    fig, axes = plt.subplots(len(res), 3, figsize=(13, 4.2 * len(res)), squeeze=False)
    for row, ((k, m), r) in zip(axes, res.items()):
        Pc, Pq = r["Sc"][:, [a, b]], r["Sq"][:, [a, b]]
        allv = np.vstack([Pc, Pq])
        pad = 0.06 * (allv.max(0) - allv.min(0))
        lo, hi = allv.min(0) - pad, allv.max(0) + pad
        scatter_groups(row[0], Pc, y)
        scatter_groups(row[1], Pq, y)
        for i in range(len(y)):
            row[2].plot([Pc[i, 0], Pq[i, 0]], [Pc[i, 1], Pq[i, 1]], color="0.6", lw=0.6, zorder=1)
        scatter_groups(row[2], Pc, y, size=30)
        scatter_groups(row[2], Pq, y, hollow=True, size=60)
        titles = ["Classical PCA", "qPCA", "Overlay (filled: PCA, open: qPCA)"]
        xl = [f"PC{pair[0]} ({evr[a]:.1%})", f"qPC{pair[0]}", f"PC{pair[0]} / qPC{pair[0]}"]
        yl = [f"PC{pair[1]} ({evr[b]:.1%})", f"qPC{pair[1]}", f"PC{pair[1]} / qPC{pair[1]}"]
        for ax, t, xlab, ylab in zip(row, titles, xl, yl):
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_title(t, loc="left", fontsize=10)
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        row[0].text(-0.28, 1.08, f"k = {k}, m = {m}", transform=row[0].transAxes,
                    fontsize=11, fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


def plot_loadings(r, k, m, out, ncols=4):
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), squeeze=False)
    for j in range(nrows * ncols):
        ax = axes[j // ncols, j % ncols]
        if j >= k:
            ax.axis("off")
            continue
        x, v = r["Wc"][j], r["Wq"][j]
        lim = 1.08 * max(np.abs(x).max(), np.abs(v).max())
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, zorder=1)
        ax.scatter(x, v, s=6, color="#0072B2", alpha=0.6, zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"PC{j + 1} vs qPC{j + 1}  |cos| = {abs(x @ v):.4f}", fontsize=9, loc="left")
        ax.set_xlabel("classical loading")
        ax.set_ylabel("quantum loading")
    fig.suptitle(f"Gene loadings, k = {k}, m = {m} (one point per gene)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--module", default="qpca_ilc730_v2")
    ap.add_argument("--configs", nargs="+", default=["6,8", "8,9"], help="k,m pairs")
    ap.add_argument("--pair", nargs=2, type=int, default=[1, 2], help="components on the axes")
    ap.add_argument("--prefix", default="fig")
    args = ap.parse_args()

    P = importlib.import_module(args.module)
    configs = [tuple(int(v) for v in c.split(",")) for c in args.configs]
    res, y, evr = run_configs(P, args.data_dir, configs)

    plot_scores(res, y, evr, args.pair, f"{args.prefix}_scores_pc{args.pair[0]}_pc{args.pair[1]}.png")
    for (k, m), r in res.items():
        plot_loadings(r, k, m, f"{args.prefix}_loadings_k{k}_m{m}.png")


if __name__ == "__main__":
    main()
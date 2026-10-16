"""
UMAP of the samples built from classical PC scores versus qPC scores.
Each (k, m) setting uses its first k components as the UMAP input.
 
Usage:
  pip install umap-learn
  python plot_umap_pc_vs_qpc.py --data-dir . --module qpca_ilc730_v2
"""
 
import argparse
import importlib
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from scipy.linalg import orthogonal_procrustes
 
from plot_scores_pc_vs_qpc import run_configs, scatter_groups
 
 
def embed(scores, n_neighbors, min_dist, seed):
    init = 10 * scores[:, :2] / np.abs(scores[:, :2]).max()  # start from the first two components
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, init=init,
                     random_state=seed).fit_transform(scores)
 
 
def align(E, ref):
    # rotation/reflection only (no scaling), so the layout is comparable but not distorted
    Ec, Rc = E - E.mean(0), ref - ref.mean(0)
    R, _ = orthogonal_procrustes(Ec, Rc)
    return Ec @ R + ref.mean(0)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--module", default="qpca_ilc730_v2")
    ap.add_argument("--configs", nargs="+", default=["6,8", "8,9"], help="k,m pairs")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fig_umap_pc_vs_qpc.png")
    args = ap.parse_args()
 
    P = importlib.import_module(args.module)
    configs = [tuple(int(v) for v in c.split(",")) for c in args.configs]
    res, y, _ = run_configs(P, args.data_dir, configs)
 
    fig, axes = plt.subplots(len(res), 2, figsize=(9.5, 4.3 * len(res)), squeeze=False)
    for row, ((k, m), r) in zip(axes, res.items()):
        Ec = embed(r["Sc"], args.n_neighbors, args.min_dist, args.seed)
        Eq = align(embed(r["Sq"], args.n_neighbors, args.min_dist, args.seed), Ec)
        lim = np.vstack([Ec, Eq])
        pad = 0.08 * (lim.max(0) - lim.min(0))
        for ax, E, title in zip(row, (Ec, Eq), (f"UMAP of classical PCs 1-{k}", f"UMAP of qPCs 1-{k}")):
            scatter_groups(ax, E, y, size=40)
            ax.set_xlim(lim[:, 0].min() - pad[0], lim[:, 0].max() + pad[0])
            ax.set_ylim(lim[:, 1].min() - pad[1], lim[:, 1].max() + pad[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_title(title, loc="left", fontsize=10)
        row[0].text(-0.1, 1.1, f"k = {k}, m = {m}", transform=row[0].transAxes,
                    fontsize=11, fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("saved", args.out)
 
 
if __name__ == "__main__":
    main()
 
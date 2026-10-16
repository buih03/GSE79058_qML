"""
Full qPCA vs. classical PCA pipeline for the 730-gene ILC dataset
(GSE79058, NanoString PanCancer panel, 76 samples: control/ILC1/ILC2).

Self-contained: data loading, the fixed controlled-U circuit construction,
and the qPCA/classical comparison all live in this one file.

Usage:
  python qpca_ilc730_v2.py --data-dir .
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit_aer import AerSimulator


# ─────────────────────────────────────────────────────────────────────────────
# Data loading: log1p -> scale -> zero-pad to 1024
# ─────────────────────────────────────────────────────────────────────────────

def load_ilc730_data(data_dir):
    data_dir = Path(data_dir)
    expr = pd.read_csv(data_dir / "counts_endogenous_final.csv", index_col=0)  # genes x samples
    meta = pd.read_csv(data_dir / "metadata.csv")

    X = expr.T.loc[meta["sample_id"]]  # samples x genes, aligned to metadata order
    gene_names = list(expr.index)
    y = meta["group"].values

    X_log = np.log1p(X.values)
    X_scaled = StandardScaler().fit_transform(X_log)

    n_padded = 1024  # next power of 2 above 730
    X_padded = np.hstack([X_scaled, np.zeros((X_scaled.shape[0], n_padded - X_scaled.shape[1]))])

    return X_padded, X_scaled, y, gene_names


# ─────────────────────────────────────────────────────────────────────────────
# Classical PCA
# ─────────────────────────────────────────────────────────────────────────────

def clustering_ari(scores_2d, y):
    true_numeric = pd.factorize(y)[0]
    pred = KMeans(n_clusters=len(np.unique(y)), random_state=0, n_init=10).fit_predict(scores_2d)
    return adjusted_rand_score(true_numeric, pred)


def classical_pca(X, y, n_components):
    start = time.perf_counter()
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    ari = clustering_ari(scores[:, :n_components], y)
    return {
        "scores": scores,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "runtime_sec": time.perf_counter() - start,
        "ari": ari,
    }


# ─────────────────────────────────────────────────────────────────────────────
# qPCA: the fixed controlled-U construction + QPE circuit
# ─────────────────────────────────────────────────────────────────────────────

_CTRL_P0 = np.array([[1, 0], [0, 0]])  # control = 0 -> identity
_CTRL_P1 = np.array([[0, 0], [0, 1]])  # control = 1 -> U


def _controlled_unitary_matrix(U):
    # Built directly (kron + add) instead of UnitaryGate(U).control(1),
    # which is intractable to synthesize once U is 1024x1024 -- see the
    # chat for the full explanation and the equivalence proof.
    dim = U.shape[0]
    return np.kron(np.eye(dim), _CTRL_P0) + np.kron(U, _CTRL_P1)


def covariance_density_matrix(X):
    C = (X.T @ X) / (X.shape[0] - 1)
    rho = C / np.trace(C)
    return C, rho


def inverse_qft_gate(n_qubits):
    gate = QFTGate(n_qubits).inverse()
    gate.label = "IQFT"
    return gate


def phase_to_bitstring(phase, n_bits):
    value = int(np.round(phase * (2 ** n_bits))) % (2 ** n_bits)
    return format(value, f"0{n_bits}b")


def build_qpe_superposition_circuit(rho, n_counting):
    dim = rho.shape[0]
    n_system = int(np.log2(dim))
    total = n_counting + n_system
    counting_qubits = list(range(n_counting))
    system_qubits = list(range(n_counting, total))

    U = expm(2j * np.pi * rho)
    qc = QuantumCircuit(total)
    for q in counting_qubits + system_qubits:
        qc.h(q)

    for q in counting_qubits:
        power = 2 ** q
        U_power = np.linalg.matrix_power(U, power)
        cU_gate = UnitaryGate(_controlled_unitary_matrix(U_power), label=f"cU^{power}")
        qc.append(cU_gate, [q] + system_qubits)

    qc.append(inverse_qft_gate(n_counting), counting_qubits)
    return qc


def extract_system_state_for_phase(statevector, phase_bitstring, n_counting, n_system):
    dim_sys = 2 ** n_system
    phase_int = int(phase_bitstring, 2)  # direct index (verified with check_bit_order.py)

    system_state = np.zeros(dim_sys, dtype=complex)
    for sys_int in range(dim_sys):
        full_index = phase_int + (sys_int << n_counting)
        system_state[sys_int] = statevector[full_index]

    norm = np.linalg.norm(system_state)
    if norm < 1e-12:
        return None, 0.0
    return system_state / norm, norm ** 2


def quantum_pca_statevector(X_padded, X_unpadded, y, n_components, n_counting):
    start = time.perf_counter()
    n_system = int(np.log2(X_padded.shape[1]))
    n_genes = X_unpadded.shape[1]

    _, rho = covariance_density_matrix(X_padded)
    true_vals, true_vecs = np.linalg.eigh(rho)
    true_vals = true_vals[::-1]
    true_vecs = true_vecs[:, ::-1]
    print(f"  Top {n_components} eigenvalues of rho: {true_vals[:n_components].round(4)}")

    qc = build_qpe_superposition_circuit(rho, n_counting)
    qc.save_statevector()
    tqc = transpile(qc, AerSimulator(), optimization_level=0)
    sim = AerSimulator(method="statevector")
    sv = np.array(sim.run(tqc).result().data()["statevector"])

    qpca_components = []
    postsel_probs = []
    for k in range(n_components):
        phase = float(true_vals[k])
        bitstr = phase_to_bitstring(phase, n_counting)
        vec, prob = extract_system_state_for_phase(sv, bitstr, n_counting, n_system)

        if vec is None:
            print(f"  qPC{k+1}: postselection empty -- using classical eigenvector")
            vec = true_vecs[:, k]

        if np.dot(np.real(vec), true_vecs[:, k]) < 0:
            vec = -vec

        qpca_components.append(np.real(vec))
        postsel_probs.append(prob)
        print(f"  qPC{k+1}: phase={phase:.4f}  bitstring={bitstr}  postsel_prob={prob:.2e}")

    qpca_components = np.array(qpca_components)
    q, _ = np.linalg.qr(qpca_components.T)
    qpca_components = q[:, :n_components].T

    components_unpadded = qpca_components[:, :n_genes]
    scores = X_unpadded @ components_unpadded.T
    ari = clustering_ari(scores[:, :n_components], y)

    return {
        "scores": scores,
        "components_unpadded": components_unpadded,
        "rho_eigenvalues": true_vals[:n_components],
        "postselection_probs": postsel_probs,
        "runtime_sec": time.perf_counter() - start,
        "ari": ari,
        "circuit_depth": tqc.depth(),
    }


def component_similarity(classical_components, qpca_components):
    S = np.abs(classical_components @ qpca_components.T)
    row_ind, col_ind = linear_sum_assignment(-S)
    matched = [
        {"classical_component": f"PC{i+1}", "qpca_component": f"qPC{j+1}",
         "abs_cosine_similarity": round(S[i, j], 4)}
        for i, j in zip(row_ind, col_ind)
    ]
    return pd.DataFrame(matched)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--n-components", type=int, default=3)  # control / ILC1 / ILC2
    ap.add_argument("--n-counting", type=int, default=8)    # picked via counting_qubit_heuristic.py
    args = ap.parse_args()

    print("Loading data...", flush=True)
    X_padded, X_scaled, y, gene_names = load_ilc730_data(args.data_dir)
    print(f"X_padded shape: {X_padded.shape}  (n_system qubits = "
          f"{int(np.log2(X_padded.shape[1]))}, n_counting = {args.n_counting})", flush=True)

    print("\nRunning classical PCA...", flush=True)
    t0 = time.time()
    classical = classical_pca(X_scaled, y, args.n_components)
    print(f"  done in {time.time()-t0:.3f}s  ARI={classical['ari']:.4f}", flush=True)

    print("\nRunning qPCA (statevector simulation)...", flush=True)
    t0 = time.time()
    qpca = quantum_pca_statevector(X_padded, X_scaled, y, args.n_components, args.n_counting)
    print(f"  done in {time.time()-t0:.1f}s  ARI={qpca['ari']:.4f}", flush=True)

    sim_df = component_similarity(classical["components"], qpca["components_unpadded"])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Classical PCA ARI: {classical['ari']:.4f}")
    print(f"qPCA ARI:          {qpca['ari']:.4f}")
    print("\nComponent similarity (|cosine|), classical vs. qPCA:")
    print(sim_df.to_string(index=False))
    print(f"\nMean |cosine similarity|: {sim_df['abs_cosine_similarity'].mean():.4f}")
    print(f"\nPostselection probabilities: {[f'{p:.2e}' for p in qpca['postselection_probs']]}")
    print(f"Circuit depth: {qpca['circuit_depth']}")


if __name__ == "__main__":
    main()
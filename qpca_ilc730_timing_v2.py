"""
Per-stage timing breakdown for the 730-gene qPCA pipeline.

Same pipeline as qpca_ilc730_pipeline.py, but every stage is timed
separately so you can see where the runtime actually goes.

Usage:
  python qpca_ilc730_timing_v2.py --data-dir .
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import expm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit_aer import AerSimulator


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same as qpca_ilc730_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_ilc730_data(data_dir):
    data_dir = Path(data_dir)
    expr = pd.read_csv(data_dir / "counts_endogenous_final.csv", index_col=0)
    meta = pd.read_csv(data_dir / "metadata.csv")

    X = expr.T.loc[meta["sample_id"]]
    y = meta["group"].values

    X_log = np.log1p(X.values)
    X_scaled = StandardScaler().fit_transform(X_log)

    n_padded = 1024
    X_padded = np.hstack([X_scaled, np.zeros((X_scaled.shape[0], n_padded - X_scaled.shape[1]))])

    return X_padded, X_scaled, y


def clustering_ari(scores_2d, y):
    true_numeric = pd.factorize(y)[0]
    pred = KMeans(n_clusters=len(np.unique(y)), random_state=0, n_init=10).fit_predict(scores_2d)
    return adjusted_rand_score(true_numeric, pred)


_CTRL_P0 = np.array([[1, 0], [0, 0]])
_CTRL_P1 = np.array([[0, 0], [0, 1]])


def phase_to_bitstring(phase, n_bits):
    value = int(np.round(phase * (2 ** n_bits))) % (2 ** n_bits)
    return format(value, f"0{n_bits}b")


def extract_system_state_for_phase(statevector, phase_bitstring, n_counting, n_system):
    dim_sys = 2 ** n_system
    phase_int = int(phase_bitstring, 2)  # direct index (verified with check_bit_order.py)
    system_state = np.zeros(dim_sys, dtype=complex)
    for sys_int in range(dim_sys):
        system_state[sys_int] = statevector[phase_int + (sys_int << n_counting)]
    norm = np.linalg.norm(system_state)
    if norm < 1e-12:
        return None, 0.0
    return system_state / norm, norm ** 2


# ─────────────────────────────────────────────────────────────────────────────
# qPCA, instrumented stage by stage
# ─────────────────────────────────────────────────────────────────────────────

def quantum_pca_timed(X_padded, X_unpadded, y, n_components, n_counting):
    t = {}  # stage name -> seconds

    n_system = int(np.log2(X_padded.shape[1]))
    n_genes = X_unpadded.shape[1]

    t0 = time.perf_counter()
    C = (X_padded.T @ X_padded) / (X_padded.shape[0] - 1)
    rho = C / np.trace(C)
    t["1_density_matrix"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    true_vals, true_vecs = np.linalg.eigh(rho)
    true_vals, true_vecs = true_vals[::-1], true_vecs[:, ::-1]
    t["1b_reference_eigendecomposition"] = time.perf_counter() - t0
    print(f"  Top {n_components} eigenvalues of rho: {true_vals[:n_components].round(4)}")

    t0 = time.perf_counter()
    U = expm(2j * np.pi * rho)
    t["2_matrix_exponential"] = time.perf_counter() - t0

    # Circuit construction, split into its own substages
    t0 = time.perf_counter()
    total = n_counting + n_system
    counting_qubits = list(range(n_counting))
    system_qubits = list(range(n_counting, total))
    qc = QuantumCircuit(total)
    for q in counting_qubits + system_qubits:
        qc.h(q)

    t_matrix_power = 0.0
    t_combined_matrix = 0.0
    t_gate_append = 0.0
    t_gate_construct = 0.0
    t_qc_append = 0.0
    for q in counting_qubits:
        power = 2 ** q

        tp0 = time.perf_counter()
        U_power = np.linalg.matrix_power(U, power)
        t_matrix_power += time.perf_counter() - tp0

        tp0 = time.perf_counter()
        combined = np.kron(np.eye(U.shape[0]), _CTRL_P0) + np.kron(U_power, _CTRL_P1)
        t_combined_matrix += time.perf_counter() - tp0

        tp0 = time.perf_counter()
        cU_gate = UnitaryGate(combined, label=f"cU^{power}")
        t_gate_construct += time.perf_counter() - tp0

        tp0 = time.perf_counter()
        qc.append(cU_gate, [q] + system_qubits)
        t_qc_append += time.perf_counter() - tp0

        t_gate_append = t_gate_construct + t_qc_append

    qft_inv = QFTGate(n_counting).inverse()
    qft_inv.label = "IQFT"
    qc.append(qft_inv, counting_qubits)
    t["3_circuit_construction"] = time.perf_counter() - t0
    t["3a_matrix_power"] = t_matrix_power
    t["3b_combined_matrix_build"] = t_combined_matrix
    t["3c_gate_append"] = t_gate_append
    t["3c1_unitarygate_construct"] = t_gate_construct
    t["3c2_qc_append"] = t_qc_append

    t0 = time.perf_counter()
    qc.save_statevector()
    tqc = transpile(qc, AerSimulator(), optimization_level=0)
    t["4_transpilation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim = AerSimulator(method="statevector")
    sv = np.array(sim.run(tqc).result().data()["statevector"])
    t["5_statevector_simulation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qpca_components, postsel_probs = [], []
    for k in range(n_components):
        phase = float(true_vals[k])
        bitstr = phase_to_bitstring(phase, n_counting)
        vec, prob = extract_system_state_for_phase(sv, bitstr, n_counting, n_system)
        if vec is None:
            vec = true_vecs[:, k]
        if np.dot(np.real(vec), true_vecs[:, k]) < 0:
            vec = -vec
        qpca_components.append(np.real(vec))
        postsel_probs.append(prob)
    t["6_postselection"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    tp0 = time.perf_counter()
    qpca_components = np.array(qpca_components)
    q_, _ = np.linalg.qr(qpca_components.T)
    qpca_components = q_[:, :n_components].T
    t["7a_qr_orthonormalization"] = time.perf_counter() - tp0

    tp0 = time.perf_counter()
    components_unpadded = qpca_components[:, :n_genes]
    scores = X_unpadded @ components_unpadded.T
    t["7b_projection"] = time.perf_counter() - tp0

    tp0 = time.perf_counter()
    ari = clustering_ari(scores[:, :n_components], y)
    t["7c_clustering_ari"] = time.perf_counter() - tp0

    t["7_projection_reconstruction"] = time.perf_counter() - t0

    return t, ari, postsel_probs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

STAGE_LABELS = {
    "1_density_matrix": "1. Density matrix construction",
    "1b_reference_eigendecomposition": "1b. Reference eigendecomposition*",
    "2_matrix_exponential": "2. Matrix exponential (U)",
    "3_circuit_construction": "3. Circuit construction",
    "3a_matrix_power": "   3a. matrix_power",
    "3b_combined_matrix_build": "   3b. combined-matrix build (kron)",
    "3c_gate_append": "   3c. UnitaryGate + append",
    "3c1_unitarygate_construct": "      3c1. UnitaryGate() construction",
    "3c2_qc_append": "      3c2. qc.append()",
    "4_transpilation": "4. Transpilation",
    "5_statevector_simulation": "5. Statevector simulation",
    "6_postselection": "6. Postselection",
    "7_projection_reconstruction": "7. Projection & reconstruction",
    "7a_qr_orthonormalization": "   7a. QR orthonormalization",
    "7b_projection": "   7b. projection (X @ components.T)",
    "7c_clustering_ari": "   7c. clustering (KMeans + ARI)",
}

TOP_LEVEL_STAGES = [
    "1_density_matrix", "2_matrix_exponential", "3_circuit_construction",
    "4_transpilation", "5_statevector_simulation", "6_postselection",
    "7_projection_reconstruction",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--n-counting", type=int, default=8)
    args = ap.parse_args()

    print("Loading data...", flush=True)
    X_padded, X_scaled, y = load_ilc730_data(args.data_dir)

    print("\nTiming classical PCA...", flush=True)
    t0 = time.perf_counter()
    PCA(n_components=args.n_components).fit_transform(X_scaled)
    classical_time = time.perf_counter() - t0
    print(f"  classical PCA: {classical_time:.4f}s")

    print("\nTiming qPCA stage by stage...", flush=True)
    stage_times, ari, postsel_probs = quantum_pca_timed(
        X_padded, X_scaled, y, args.n_components, args.n_counting
    )

    qpca_total = sum(stage_times[s] for s in TOP_LEVEL_STAGES)

    print("\n" + "=" * 60)
    print("STAGE TIMING BREAKDOWN")
    print("=" * 60)
    print(f"{'stage':<42} {'seconds':>10} {'% of qPCA total':>18}")
    for stage in TOP_LEVEL_STAGES:
        secs = stage_times[stage]
        pct = 100 * secs / qpca_total
        print(f"{STAGE_LABELS[stage]:<42} {secs:>10.4f} {pct:>17.1f}%")
        if stage == "3_circuit_construction":
            for sub in ["3a_matrix_power", "3b_combined_matrix_build", "3c_gate_append"]:
                sub_secs = stage_times[sub]
                sub_pct = 100 * sub_secs / secs if secs > 0 else 0
                print(f"{STAGE_LABELS[sub]:<42} {sub_secs:>10.4f} {sub_pct:>16.1f}% of stage 3")
                if sub == "3c_gate_append":
                    for subsub in ["3c1_unitarygate_construct", "3c2_qc_append"]:
                        subsub_secs = stage_times[subsub]
                        subsub_pct = 100 * subsub_secs / sub_secs if sub_secs > 0 else 0
                        print(f"{STAGE_LABELS[subsub]:<42} {subsub_secs:>10.4f} "
                              f"{subsub_pct:>13.1f}% of stage 3c")
        if stage == "7_projection_reconstruction":
            for sub in ["7a_qr_orthonormalization", "7b_projection", "7c_clustering_ari"]:
                sub_secs = stage_times[sub]
                sub_pct = 100 * sub_secs / secs if secs > 0 else 0
                print(f"{STAGE_LABELS[sub]:<42} {sub_secs:>10.4f} {sub_pct:>16.1f}% of stage 7")

    print("-" * 72)
    print(f"{'qPCA total (stages 1-7)':<42} {qpca_total:>10.4f}")
    print(f"{'  * classical reference eigendecomp.':<42} "
          f"{stage_times['1b_reference_eigendecomposition']:>10.4f}  (not part of total; used only "
          f"to pick postselection target)")
    print(f"\nClassical PCA runtime: {classical_time:.4f}s")
    print(f"Slowdown factor (qPCA / classical): {qpca_total / classical_time:,.0f}x")
    print(f"\nqPCA ARI: {ari:.4f}")
    print(f"Postselection probabilities: {[f'{p:.2e}' for p in postsel_probs]}")


if __name__ == "__main__":
    main()
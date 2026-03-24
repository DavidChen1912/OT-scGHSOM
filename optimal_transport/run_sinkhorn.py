import os
import argparse
import numpy as np
import pandas as pd


# ============================================================
# Sinkhorn (stable version)
# ============================================================
def sinkhorn(a, b, C, epsilon=0.05, max_iter=1000, tol=1e-9):
    K = np.exp(-C / epsilon)
    K[K < 1e-300] = 1e-300

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(max_iter):
        u_prev = u.copy()

        u = a / (K @ v)
        v = b / (K.T @ u)

        if np.linalg.norm(u - u_prev, 1) < tol:
            break

    P = np.diag(u) @ K @ np.diag(v)
    return P


# ============================================================
# Main
# ============================================================
def run_sinkhorn(data_name, tau1, tau2, source, target, epsilon, patient_col):

    # ============================================================
    # Load cost matrix
    # ============================================================
    cost_path = os.path.join(
        "optimal_transport",
        "cost_matrix",
        f"{data_name}-{tau1}-{tau2}_cost_matrix.npy"
    )

    if not os.path.exists(cost_path):
        raise FileNotFoundError(cost_path)

    C = np.load(cost_path)

    # ============================================================
    # Load distribution
    # ============================================================
    dist_path = os.path.join(
        "optimal_transport",
        "data",
        f"{data_name}-{tau1}-{tau2}_distribution.csv"
    )

    if not os.path.exists(dist_path):
        raise FileNotFoundError(dist_path)

    df = pd.read_csv(dist_path)

    # 🔥 改這裡
    if patient_col not in df.columns:
        raise ValueError(f"Column '{patient_col}' not found in distribution file")

    # ============================================================
    # valid patients
    # ============================================================
    patients_source = set(df[df["time"] == source][patient_col].unique())
    patients_target = set(df[df["time"] == target][patient_col].unique())

    valid_patients = sorted(list(patients_source & patients_target))

    if len(valid_patients) == 0:
        raise ValueError("No patients have both source and target time")

    print(f"[INFO] valid patients: {len(valid_patients)}")

    # ============================================================
    # Output folder
    # ============================================================
    output_dir = os.path.join(
        "optimal_transport",
        "result",
        f"{data_name}-{tau1}-{tau2}_{source}_to_{target}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Loop patients
    # ============================================================
    for patient in valid_patients:

        print(f"[RUN] {patient}")

        df_p = df[df[patient_col] == patient]

        df_source = df_p[df_p["time"] == source].copy()
        df_target = df_p[df_p["time"] == target].copy()

        df_source = df_source.sort_values("cluster_number")
        df_target = df_target.sort_values("cluster_number")

        a = df_source["prob"].values
        b = df_target["prob"].values
        clusters = df_source["cluster_number"].values

        # sanity check
        if not np.isclose(a.sum(), 1):
            print(f"[SKIP] {patient} source not valid")
            continue

        if not np.isclose(b.sum(), 1):
            print(f"[SKIP] {patient} target not valid")
            continue

        # Sinkhorn
        P = sinkhorn(a, b, C, epsilon=epsilon)

        total_cells = df_source["count"].sum()
        P_count = P * total_cells

        # ========================================================
        # Output
        # ========================================================
        prob_npy = os.path.join(output_dir, f"{patient}_prob.npy")
        prob_csv = os.path.join(output_dir, f"{patient}_prob.csv")

        count_npy = os.path.join(output_dir, f"{patient}_count.npy")
        count_csv = os.path.join(output_dir, f"{patient}_count.csv")

        np.save(prob_npy, P)
        np.save(count_npy, P_count)

        pd.DataFrame(P, index=clusters, columns=clusters).to_csv(prob_csv)
        pd.DataFrame(P_count, index=clusters, columns=clusters).to_csv(count_csv)

    print(f"[DONE] Results saved to: {output_dir}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--tau1", required=True)
    parser.add_argument("--tau2", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)

    # ⭐ NEW
    parser.add_argument("--patient", required=True)

    args = parser.parse_args()

    run_sinkhorn(
        data_name=args.data,
        tau1=args.tau1,
        tau2=args.tau2,
        source=args.source,
        target=args.target,
        epsilon=args.epsilon,
        patient_col=args.patient
    )

# python optimal_transport/run_sinkhorn.py --data=CART_0320 --tau1=0.5 --tau2=0.5 --patient=patient --source=day0 --target=day7 --epsilon=0.05
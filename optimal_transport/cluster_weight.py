import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def get_project_root():
    return Path(__file__).resolve().parents[1]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--tau1", required=True)
    parser.add_argument("--tau2", required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--index", required=False)
    parser.add_argument("--patient", required=False)  # ⭐ NEW

    args = parser.parse_args()

    data = args.data
    tau1 = args.tau1
    tau2 = args.tau2
    time_col = args.time
    index_col = args.index
    patient_col = args.patient  # ⭐ NEW

    root = get_project_root()
    os.chdir(root)

    # ============================================================
    # Step 1: 取得 marker 名稱
    # ============================================================

    raw_path = root / "raw-data" / f"{data}.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Cannot find raw data: {raw_path}")

    raw_df = pd.read_csv(raw_path)

    columns = list(raw_df.columns)

    if time_col in columns:
        columns.remove(time_col)

    if index_col and index_col in columns:
        columns.remove(index_col)

    # ⭐ NEW: drop patient column
    if patient_col and patient_col in columns:
        columns.remove(patient_col)

    markers = columns

    print(f"[INFO] markers detected: {len(markers)}")


    # ============================================================
    # Step 2: 讀取 clustered dataset
    # ============================================================

    cluster_path = (
        root
        / "applications"
        / f"{data}-{tau1}-{tau2}"
        / "data"
        / f"{data}_with_clustered_label-{tau1}-{tau2}.csv"
    )

    if not cluster_path.exists():
        raise FileNotFoundError(f"Cannot find clustered data: {cluster_path}")

    print(f"[LOAD] {cluster_path}")

    df = pd.read_csv(cluster_path)

    required_cols = markers + ["cluster_number"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required_cols]


    # ============================================================
    # Step 3: cluster mean matrix
    # ============================================================

    cluster_means = df.groupby("cluster_number")[markers].mean()
    clusters = cluster_means.index.tolist()
    num_clusters = len(clusters)

    epsilon = 1e-8

    weight_rows = []

    # ============================================================
    # Step 4: 計算每個 cluster 的 feature weight
    # ============================================================

    for c in clusters:

        sub_df = df[df["cluster_number"] == c]

        diff_scores = {}

        for g in markers:

            # ---------- σ_I ----------
            cluster_mean = sub_df[g].mean()
            sigma_I = np.sqrt(((sub_df[g] - cluster_mean) ** 2).sum() / len(sub_df))

            # ---------- σ_B ----------
            m_c = cluster_means.loc[c, g]

            others = [x for x in clusters if x != c]
            m_others = cluster_means.loc[others, g]

            sigma_B = np.sqrt(((m_c - m_others) ** 2).sum() / len(others))

            diff_scores[g] = sigma_B - sigma_I


        # ========================================================
        # Step 5: shift + epsilon
        # ========================================================

        diffs = np.array(list(diff_scores.values()))
        min_diff = diffs.min()

        shifted = {g: (diff_scores[g] - min_diff + epsilon) for g in markers}


        # ========================================================
        # Step 6: normalize
        # ========================================================

        total = sum(shifted.values())

        weights = {g: shifted[g] / total for g in markers}

        row = {"cluster_number": c}
        row.update(weights)

        weight_rows.append(row)


    # ============================================================
    # Step 7: output
    # ============================================================

    weight_df = pd.DataFrame(weight_rows)
    weight_df = weight_df.sort_values("cluster_number")

    output_dir = root / "optimal_transport" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{data}-{tau1}-{tau2}_cluster_weight.csv"

    weight_df.to_csv(output_path, index=False)

    print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()

# python optimal_transport/cluster_weight.py --data=CART_0320 --tau1=0.5 --tau2=0.5 --time=time --index=ID --patient=patient
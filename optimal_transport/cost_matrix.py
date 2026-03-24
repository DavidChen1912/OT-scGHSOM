import os
import argparse
import numpy as np
import pandas as pd


# ============================================================
# Utility: Load CSV and align clusters
# ============================================================
def load_feature_data(center_path, weight_path):
    center_df = pd.read_csv(center_path)
    weight_df = pd.read_csv(weight_path)

    # 確保 cluster_number 對齊
    center_df = center_df.sort_values("cluster_number").reset_index(drop=True)
    weight_df = weight_df.sort_values("cluster_number").reset_index(drop=True)

    if not (center_df["cluster_number"].values == weight_df["cluster_number"].values).all():
        raise ValueError("cluster_number mismatch between center and weight")

    clusters = center_df["cluster_number"].values

    # 拿 feature（去掉第一欄）
    center_feat = center_df.drop(columns=["cluster_number"]).values
    weight_feat = weight_df.drop(columns=["cluster_number"]).values

    # weighted feature
    z = center_feat * weight_feat

    return clusters, z


# ============================================================
# Utility: Hierarchy distance (NEW VERSION)
# ============================================================
def compute_hierarchy_matrix(clusters, hierarchy_df):
    """
    hierarchy_df columns:
        cluster_number
        layer
        sibling_group
    """

    n = len(clusters)
    H = np.zeros((n, n))

    # --------------------------------------------------------
    # 建立 cluster -> layer / sibling_group lookup table
    # --------------------------------------------------------
    layer_map = {}
    group_map = {}

    for _, row in hierarchy_df.iterrows():

        c = int(row["cluster_number"])
        layer_map[c] = int(row["layer"])
        group_map[c] = str(row["sibling_group"])

    # --------------------------------------------------------
    # 計算 hierarchy distance
    # --------------------------------------------------------
    for i in range(n):
        for j in range(n):

            ci = int(clusters[i])
            cj = int(clusters[j])

            # 同一個 cluster
            if ci == cj:
                H[i, j] = 0
                continue

            layer_i = layer_map.get(ci, 1)
            layer_j = layer_map.get(cj, 1)

            group_i = group_map.get(ci, "NONE")
            group_j = group_map.get(cj, "NONE")

            # ------------------------------------------------
            # CASE 1: siblings
            # 條件：
            # 1. layer 相同
            # 2. sibling_group 相同
            # 3. group 不能是 NONE
            # ------------------------------------------------
            if (
                layer_i == layer_j
                and group_i == group_j
                and group_i != "NONE"
            ):
                dist = 1

            # ------------------------------------------------
            # CASE 2: 其他情況
            # ------------------------------------------------
            else:
                dist = layer_i + layer_j - 1

            H[i, j] = dist

    # --------------------------------------------------------
    # normalize hierarchy distance
    # --------------------------------------------------------
    max_val = H.max()

    if max_val > 0:
        H = H / max_val

    return H


# ============================================================
# Utility: Feature distance (Euclidean)
# ============================================================
def compute_feature_matrix(z):

    n = z.shape[0]
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(z[i] - z[j])

    return D


# ============================================================
# Main
# ============================================================
def compute_cost_matrix(data_name, tau1, tau2, lambda_):

    # ============================================================
    # Paths
    # ============================================================
    base_data_path = os.path.join("optimal_transport", "data")
    hierarchy_path = os.path.join("optimal_transport", "hierarchy")

    center_path = os.path.join(
        base_data_path,
        f"{data_name}-{tau1}-{tau2}_cluster_center.csv"
    )

    weight_path = os.path.join(
        base_data_path,
        f"{data_name}-{tau1}-{tau2}_cluster_weight.csv"
    )

    # ------------------------------------------------------------
    # NEW: hierarchy 改讀 CSV
    # ------------------------------------------------------------
    hierarchy_csv = os.path.join(
        hierarchy_path,
        f"{data_name}_cluster_hierarchy-{tau1}-{tau2}.csv"
    )

    if not os.path.exists(center_path):
        raise FileNotFoundError(center_path)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(weight_path)

    if not os.path.exists(hierarchy_csv):
        raise FileNotFoundError(hierarchy_csv)

    # ============================================================
    # Load feature
    # ============================================================
    clusters, z = load_feature_data(center_path, weight_path)

    # ============================================================
    # Feature distance
    # ============================================================
    D_feat = compute_feature_matrix(z)

    # ============================================================
    # Hierarchy (NEW)
    # ============================================================
    hierarchy_df = pd.read_csv(hierarchy_csv)

    D_hier = compute_hierarchy_matrix(clusters, hierarchy_df)

    # ============================================================
    # Combine cost
    # ============================================================
    C = D_feat + lambda_ * D_hier

    # 確保對角線為0（安全）
    np.fill_diagonal(C, 0.0)

    # ============================================================
    # Normalize cost (VERY IMPORTANT for Sinkhorn stability)
    # ============================================================
    max_val = C.max()

    if max_val > 0:
        C = C / max_val

    # ============================================================
    # Output
    # ============================================================
    output_dir = os.path.join("optimal_transport", "cost_matrix")
    os.makedirs(output_dir, exist_ok=True)

    npy_path = os.path.join(
        output_dir,
        f"{data_name}-{tau1}-{tau2}_cost_matrix.npy"
    )

    csv_path = os.path.join(
        output_dir,
        f"{data_name}-{tau1}-{tau2}_cost_matrix.csv"
    )

    # npy（給程式）
    np.save(npy_path, C)

    # csv（給人看）
    df = pd.DataFrame(C, index=clusters, columns=clusters)
    df.to_csv(csv_path)

    print(f"Saved npy: {npy_path}")
    print(f"Saved csv: {csv_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--tau1", required=True)
    parser.add_argument("--tau2", required=True)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.3)

    args = parser.parse_args()

    compute_cost_matrix(
        data_name=args.data,
        tau1=args.tau1,
        tau2=args.tau2,
        lambda_=args.lambda_
    )

# python optimal_transport/cost_matrix.py --data=CART_0320 --tau1=0.5 --tau2=0.5 --lambda=0.3
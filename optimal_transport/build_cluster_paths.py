import argparse
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data(data, tau1, tau2):

    folder = PROJECT_ROOT / "applications" / f"{data}-{tau1}-{tau2}" / "data"
    filename = f"{data}_with_clustered_label-{tau1}-{tau2}.csv"
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    print("Loading:", path)

    df = pd.read_csv(path)

    return df


def get_hierarchy_columns(df):

    cols = [c for c in df.columns if c.startswith("clusterL")]

    if not cols:
        raise ValueError("No clusterL columns found.")

    cols = sorted(
        cols,
        key=lambda x: int(x.replace("clusterL", ""))
    )

    print("Hierarchy columns:", cols)

    return cols


def build_cluster_table(df):

    hierarchy_cols = get_hierarchy_columns(df)

    if "cluster_number" not in df.columns:
        raise ValueError("cluster_number column missing")

    # 每個 cluster 只取一筆代表 cell
    df_unique = df.drop_duplicates(subset=["cluster_number"]).copy()

    records = []

    for _, row in df_unique.iterrows():

        cluster = int(row["cluster_number"])

        path = []

        for col in hierarchy_cols:

            val = row[col]

            if pd.isna(val):
                break

            val = str(val).strip()

            if val == "":
                break

            path.append(val)

        layer = len(path)

        # prefix = 去掉最後一層
        prefix = tuple(path[:-1]) if layer > 1 else None

        records.append({
            "cluster_number": cluster,
            "layer": layer,
            "prefix": prefix
        })

    df_clusters = pd.DataFrame(records)

    # -------------------------
    # 建立 sibling group
    # -------------------------

    group_map = {}
    group_id = 1
    sibling_groups = []

    for _, row in df_clusters.iterrows():

        layer = row["layer"]
        prefix = row["prefix"]

        # 第一層 cluster 不可能有 siblings
        if layer == 1:
            sibling_groups.append("NONE")
            continue

        key = (layer, prefix)

        if key not in group_map:
            group_map[key] = f"G{group_id}"
            group_id += 1

        sibling_groups.append(group_map[key])

    df_clusters["sibling_group"] = sibling_groups

    # -------------------------
    # 如果 group size = 1 → NONE
    # -------------------------

    counts = df_clusters["sibling_group"].value_counts()

    df_clusters["sibling_group"] = df_clusters["sibling_group"].apply(
        lambda g: g if g != "NONE" and counts[g] > 1 else "NONE"
    )

    return df_clusters[["cluster_number", "layer", "sibling_group"]]


def save_csv(df_clusters, data, tau1, tau2):

    out_dir = PROJECT_ROOT / "optimal_transport" / "hierarchy"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{data}_cluster_hierarchy-{tau1}-{tau2}.csv"

    df_clusters.to_csv(out_path, index=False)

    print("\nCluster hierarchy saved →", out_path)
    print("Total clusters:", len(df_clusters))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--tau1", required=True)
    parser.add_argument("--tau2", required=True)

    args = parser.parse_args()

    df = load_data(args.data, args.tau1, args.tau2)

    df_clusters = build_cluster_table(df)

    save_csv(df_clusters, args.data, args.tau1, args.tau2)


if __name__ == "__main__":
    main()

# python optimal_transport/build_cluster_paths.py --data=CART_0320 --tau1=0.5 --tau2=0.5
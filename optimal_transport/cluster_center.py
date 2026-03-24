import argparse
import os
from pathlib import Path
import pandas as pd


def get_project_root():
    """
    Return project root (two levels up from this script)
    optimal_transport/cluster_center.py -> project root
    """
    return Path(__file__).resolve().parents[1]


def main():

    # =========================================================
    # CLI
    # =========================================================
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="dataset name")
    parser.add_argument("--tau1", required=True, help="tau1 value")
    parser.add_argument("--tau2", required=True, help="tau2 value")
    parser.add_argument("--time", required=True, help="time column name")
    parser.add_argument("--index", required=False, help="index column name")
    parser.add_argument("--patient", required=False, help="patient column name")  # ⭐ NEW

    args = parser.parse_args()

    data = args.data
    tau1 = args.tau1
    tau2 = args.tau2
    time_col = args.time
    index_col = args.index
    patient_col = args.patient  # ⭐ NEW


    # =========================================================
    # Step 1: project root
    # =========================================================
    root = get_project_root()
    os.chdir(root)

    print(f"Project root: {root}")


    # =========================================================
    # Step 2: read raw data to get marker names
    # =========================================================
    raw_path = root / "raw-data" / f"{data}.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    raw_df = pd.read_csv(raw_path)

    columns = list(raw_df.columns)

    # drop time column
    if time_col in columns:
        columns.remove(time_col)

    # drop index column
    if index_col and index_col in columns:
        columns.remove(index_col)

    # drop patient column ⭐ NEW
    if patient_col and patient_col in columns:
        columns.remove(patient_col)

    markers = columns

    print(f"Detected {len(markers)} markers")


    # =========================================================
    # Step 3: find clustered dataset
    # =========================================================
    cluster_path = (
        root
        / "applications"
        / f"{data}-{tau1}-{tau2}"
        / "data"
        / f"{data}_with_clustered_label-{tau1}-{tau2}.csv"
    )

    if not cluster_path.exists():
        raise FileNotFoundError(f"Clustered dataset not found: {cluster_path}")

    print(f"Loading clustered data: {cluster_path}")

    df = pd.read_csv(cluster_path)


    # =========================================================
    # Step 4: filter columns
    # =========================================================
    required_columns = markers + ["cluster_number"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column missing in clustered data: {col}")

    df = df[required_columns]


    # =========================================================
    # Step 5: compute cluster centers
    # =========================================================
    centers = (
        df.groupby("cluster_number")[markers]
        .mean()
        .reset_index()
        .sort_values("cluster_number")
    )


    # =========================================================
    # Step 6: output
    # =========================================================
    output_dir = root / "optimal_transport" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{data}-{tau1}-{tau2}_cluster_center.csv"

    centers.to_csv(output_path, index=False)

    print("Cluster centers saved:")
    print(output_path)


if __name__ == "__main__":
    main()

# python optimal_transport/cluster_center.py --data=CART_0320 --tau1=0.5 --tau2=0.5 --time=time --index=ID --patient=patient
import os
import argparse
import pandas as pd


def get_distribution(data_name, tau1, tau2, time_col, patient_col=None):
    # ============================================================
    # 1. Build input path
    # ============================================================
    input_path = os.path.join(
        "applications",
        f"{data_name}-{tau1}-{tau2}",
        "data",
        f"{data_name}_with_clustered_label-{tau1}-{tau2}.csv"
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # ============================================================
    # 2. Check required columns
    # ============================================================
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in dataset")

    if "cluster_number" not in df.columns:
        raise ValueError("Column 'cluster_number' not found in dataset")

    if patient_col and patient_col not in df.columns:
        raise ValueError(f"Patient column '{patient_col}' not found in dataset")

    # ============================================================
    # 3. Ensure cluster_number is numeric
    # ============================================================
    df["cluster_number"] = pd.to_numeric(df["cluster_number"])

    # ============================================================
    # 4. Get global cluster list
    # ============================================================
    all_clusters = sorted(df["cluster_number"].unique())

    # ============================================================
    # ⭐ 分兩種模式：有沒有 patient
    # ============================================================

    if patient_col:
        # ========================================================
        # Multi-patient mode
        # ========================================================

        # Groupby
        grouped = (
            df.groupby([patient_col, time_col, "cluster_number"])
            .size()
            .reset_index(name="count")
        )

        # ========================================================
        # ⭐ 修正：只針對「存在的 (patient, time)」補 cluster
        # ========================================================
        existing_pairs = df[[patient_col, time_col]].drop_duplicates()

        full_index = pd.MultiIndex.from_frame(
            existing_pairs.merge(
                pd.DataFrame({"cluster_number": all_clusters}),
                how="cross"
            )
        )

        grouped = grouped.set_index([patient_col, time_col, "cluster_number"])
        grouped = grouped.reindex(full_index, fill_value=0).reset_index()

        # Normalize（每個 patient + time）
        grouped["prob"] = grouped.groupby([patient_col, time_col])["count"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )

        # Sorting
        grouped = grouped.sort_values(by=[patient_col, time_col, "cluster_number"])

        # Rename
        grouped = grouped.rename(columns={
            patient_col: "patient",
            time_col: "time"
        })

    else:
        # ========================================================
        # 原本 single-patient mode（完全保留）
        # ========================================================

        grouped = (
            df.groupby([time_col, "cluster_number"])
            .size()
            .reset_index(name="count")
        )

        all_times = df[time_col].drop_duplicates()

        full_index = pd.MultiIndex.from_product(
            [all_times, all_clusters],
            names=[time_col, "cluster_number"]
        )

        grouped = grouped.set_index([time_col, "cluster_number"])
        grouped = grouped.reindex(full_index, fill_value=0).reset_index()

        grouped["prob"] = grouped.groupby(time_col)["count"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )

        grouped = grouped.sort_values(by=[time_col, "cluster_number"])

        grouped = grouped.rename(columns={time_col: "time"})

    # ============================================================
    # 9. Output
    # ============================================================
    output_dir = os.path.join("optimal_transport", "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{data_name}-{tau1}-{tau2}_distribution.csv"
    )

    grouped.to_csv(output_path, index=False)

    print(f"Distribution file saved to: {output_path}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--tau1", required=True)
    parser.add_argument("--tau2", required=True)
    parser.add_argument("--time", required=True)
    parser.add_argument("--patient", required=False)

    args = parser.parse_args()

    get_distribution(
        data_name=args.data,
        tau1=args.tau1,
        tau2=args.tau2,
        time_col=args.time,
        patient_col=args.patient
    )

# python optimal_transport/get_distribution.py --data=CART_0320 --tau1=0.5 --tau2=0.5 --time=time --patient=patient
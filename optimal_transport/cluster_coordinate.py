import os
import argparse
import pandas as pd
from fractions import Fraction


def fraction_to_float(x):
    """將 '1/4' 這種字串轉成 float"""
    try:
        if isinstance(x, str) and "/" in x:
            return float(Fraction(x))
        return float(x)
    except Exception:
        return None  # 或你要 raise error 也可以


def cluster_coordinate(data, tau1, tau2):
    # ============================================================
    # 1. 路徑
    # ============================================================
    folder = f"{data}-{tau1}-{tau2}"

    input_path = os.path.join(
        "applications",
        folder,
        "data",
        f"{data}_with_clustered_label-{tau1}-{tau2}.csv"
    )

    output_dir = os.path.join("optimal_transport", "data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{data}-{tau1}-{tau2}_cluster_coordinate.csv"
    )

    # ============================================================
    # 2. 讀資料
    # ============================================================
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 找不到檔案：{input_path}")

    df = pd.read_csv(input_path)

    # ============================================================
    # 3. 檢查欄位
    # ============================================================
    required_cols = ["cluster_number", "point_x", "point_y"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ 缺少欄位：{col}")

    # ============================================================
    # ⭐ 4. 轉換座標（重點）
    # ============================================================
    df["point_x"] = df["point_x"].apply(fraction_to_float)
    df["point_y"] = df["point_y"].apply(fraction_to_float)

    # ============================================================
    # 5. 每個 cluster 唯一座標
    # ============================================================
    cluster_df = (
        df.drop_duplicates(subset=["cluster_number"])[
            ["cluster_number", "point_x", "point_y"]
        ]
        .sort_values("cluster_number")
        .reset_index(drop=True)
    )

    # ============================================================
    # 6. 輸出
    # ============================================================
    cluster_df.to_csv(output_path, index=False)

    print(f"✅ 已輸出 cluster 座標：{output_path}")
    print(f"📊 cluster 數量：{len(cluster_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cluster coordinates for OT visualization")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tau1", type=float, required=True)
    parser.add_argument("--tau2", type=float, required=True)

    args = parser.parse_args()
    cluster_coordinate(args.data, args.tau1, args.tau2)

# python optimal_transport/cluster_coordinate.py --data=CART_0320 --tau1=0.5 --tau2=0.5
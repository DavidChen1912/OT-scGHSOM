import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import math
import argparse
import os

# ========================================
# 解析參數
# ========================================
parser = argparse.ArgumentParser(description='Clustering evaluation script')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--tau1', type=float, required=True)
parser.add_argument('--tau2', type=float, required=True)
parser.add_argument('--label', type=str, default=None)     # optional label 欄位
parser.add_argument('--index', type=str, default=None)     # optional index
args = parser.parse_args()

prefix = args.name
t1 = args.tau1
t2 = args.tau2
label_col = args.label
index_col = args.index

file = f"{prefix}-{t1}-{t2}"

# ========================================
# 讀取分群後的資料（GHSOM clustering result）
# ========================================
cluster_path = f'./applications/{file}/data/{prefix}_with_clustered_label-{t1}-{t2}.csv'
df_cluster = pd.read_csv(cluster_path)

# x_y_label（cluster id）
cluster_label = df_cluster['x_y_label']

# ========================================
# 讀取 Raw Data（算內部指標用）
# ========================================
raw_path = f'./raw-data/{prefix}.csv'
df_raw = pd.read_csv(raw_path)

# ========================================
# 清理 features（排除 index 與 label）
# ========================================
exclude_cols = []

if index_col is not None and index_col in df_raw.columns:
    exclude_cols.append(index_col)

if label_col is not None and label_col in df_raw.columns:
    exclude_cols.append(label_col)

sample = df_raw.drop(columns=exclude_cols, errors='ignore')

# ============================
# 第二道保險：只保留數值欄位
# ============================
sample = sample.select_dtypes(include=[np.number])

# ========================================
# 計算 Leaf Number
# ========================================
leaf_number = df_cluster['x_y_label'].nunique()

# ========================================
# 外部指標 ARI / NMI
# ========================================
if label_col is None or label_col not in df_raw.columns:
    ARI = "NA"
    NMI = "NA"
else:
    true_label = df_raw[label_col]
    if pd.api.types.is_numeric_dtype(true_label):
        true_label = true_label.fillna(-1)
    else:
        true_label = true_label.fillna("Unknown")


    if len(true_label) != len(cluster_label):
        raise ValueError("Label length does not match clustering result.")

    ARI = round(adjusted_rand_score(true_label, cluster_label), 3)
    NMI = round(normalized_mutual_info_score(true_label, cluster_label), 3)

# ========================================
# 內部指標 DB / CH
# ========================================
DB = round(davies_bouldin_score(sample, cluster_label), 3)
CH_raw = calinski_harabasz_score(sample, cluster_label)
CH = round(math.log10(CH_raw), 3)

# ========================================
# 儲存結果到 Result folder
# ========================================
os.makedirs("Result", exist_ok=True)
output_path = f"Result/{prefix}_result.csv"

df_out = pd.DataFrame({
    "CH": [CH],
    "DB": [DB],
    "ARI": [ARI],
    "NMI": [NMI],
    "Leaf_Number": [leaf_number]
})

df_out.to_csv(output_path, index=False)

# ========================================
# Print 結果（維持舊版行為）
# ========================================
print("Internal---")
print(f"CH Score (log10): {CH}")
print(f"DB Score: {DB}")

print("External---")
print(f"ARI Score: {ARI}")
print(f"NMI Score: {NMI}")

print("Leaf_Number:", leaf_number)
print(f"[OK] Result saved at {output_path}")






import numpy as np
import pandas as pd
from pandas import ExcelWriter
from fractions import Fraction
import argparse
import get_ghsom_dim

# ========================================
# Argument parsing
# ========================================
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--tau1', type=float, default=0.1)
parser.add_argument('--tau2', type=float, default=0.01)
parser.add_argument('--index', type=str, default=None)
parser.add_argument('--label', type=str, default=None)  # ⭐ optional, 不傳也沒差

args = parser.parse_args()

prefix = args.name
t1 = args.tau1
t2 = args.tau2
index_col_name = args.index
label_col_name = args.label

file = f'{prefix}-{t1}-{t2}'

# ========================================
# Load GHSOM structure info
# ========================================
layers, max_layer, number_of_digits = get_ghsom_dim.layers(file)

# ========================================
# Load raw data
# ========================================
df_source = pd.read_csv(f'./raw-data/{prefix}.csv', encoding='utf-8')

# ==========================================================
# ⭐ Robust mean / median (ONLY change)
#   1) 排除使用者指定的 index / label 欄位（若存在）
#   2) 僅針對 numeric 欄位計算
# ==========================================================
exclude_cols = set()

if index_col_name is not None and index_col_name in df_source.columns:
    exclude_cols.add(index_col_name)

if label_col_name is not None and label_col_name in df_source.columns:
    exclude_cols.add(label_col_name)

# 第一道保險：先排除 metadata 欄位
candidate = df_source.drop(columns=list(exclude_cols), errors='ignore')

# 第二道保險：只選 numeric 欄位
numeric_cols = candidate.select_dtypes(include=[np.number]).columns

if len(numeric_cols) == 0:
    mean = pd.Series(np.nan, index=df_source.index)
    median = pd.Series(np.nan, index=df_source.index)
else:
    mean = candidate[numeric_cols].mean(axis=1)
    median = candidate[numeric_cols].median(axis=1)

# ========================================
# Attach computed statistics
# ========================================
df_source['mean'] = mean
df_source['median'] = median

# ========================================
# Initialize cluster-related columns
# ========================================
df_source['clustered_label'] = np.nan
df_source['x_y_label'] = np.nan

for i in range(1, max_layer + 1):
    df_source['clusterL' + str(i)] = np.nan

# ========================================
# Utility functions
# ========================================
def get_cluster_flag(text_file):
    flags = [i for i, x in enumerate(text_file) if x == '$POS_X']
    flags.append(len(text_file) + 1)
    return flags

def format_cluster_info_to_dict(
    unit_file_name,
    source_data,
    saved_data_type=None,
    structure_type=None,
    parent_name=None,
    parent_file_position=None,
    parent_clustered_string=None,
    x_y_clustered_string=None
):
    Groups_info = []

    unit_file_path = (
        f'./applications/{file}/GHSOM/output/{file}/'
        + unit_file_name + '.unit'
    )
    print(unit_file_path)

    text_file = open(unit_file_path).read().split()
    flag = get_cluster_flag(text_file)

    if 'lvl' in unit_file_name:
        layer_index = int(unit_file_name.split('lvl')[1][0])
    else:
        layer_index = 1

    XDIM = text_file[text_file.index('$XDIM') + 1]
    YDIM = text_file[text_file.index('$YDIM') + 1]
    map_size = int(XDIM) * int(YDIM)

    if parent_name is None:
        parent_name = 'Root'
    else:
        parent_name = str(parent_name) + '-' + str(parent_file_position)

    if x_y_clustered_string is None:
        x_y_clustered_string = ''

    if parent_clustered_string is None:
        parent_clustered_string = ''

    for i, map_index in zip(range(len(flag) - 1), range(map_size)):
        currentSection = text_file[flag[i]:flag[i + 1]]

        x_position = currentSection[currentSection.index('$POS_X') + 1]
        y_position = currentSection[currentSection.index('$POS_Y') + 1]

        group_position = x_position + y_position

        group_data_index = (
            currentSection[
                currentSection.index('$MAPPED_VECS') + 1:
                currentSection.index('$MAPPED_VECS_DIST')
            ]
            if '$MAPPED_VECS' in currentSection else []
        )

        sub_map_file_name = (
            currentSection[currentSection.index('$URL_MAPPED_SOMS') + 1]
            if '$URL_MAPPED_SOMS' in currentSection else 'None'
        )

        map_index_int = map_index + 1
        digit = 1
        while (map_index_int // 10) != 0:
            digit += 1
            map_index_int //= 10

        zero = ''
        for _ in range(number_of_digits[layer_index - 1] - digit):
            zero += '0'

        cluster_string = (
            str(parent_clustered_string)
            + str(XDIM) + ';'
            + str(YDIM) + ';'
            + x_position + ';'
            + y_position + ';'
        )

        x_y_string = str(x_y_clustered_string) + '-' + x_position + 'x' + y_position

        mapped_idx = np.array(group_data_index, dtype='int64')
        current_group_source = df_source.iloc[mapped_idx, :]

        current_group_statistic_info = current_group_source.describe().to_dict()

        if sub_map_file_name != 'None':
            format_cluster_info_to_dict(
                sub_map_file_name,
                source_data,
                saved_data_type,
                structure_type,
                unit_file_name,
                group_position,
                cluster_string,
                x_y_string
            )
            leaf_node = 0
        else:
            leaf_node = 1
            dimension_list = []

            df_source.loc[mapped_idx, 'clustered_label'] = cluster_string
            df_source.loc[mapped_idx, 'x_y_label'] = x_y_string

            cluster_string = cluster_string.strip(';')
            clusters_list = cluster_string.split(';')
            levels = x_y_string.split('-')

            for j in range(0, len(clusters_list), 4):
                dimension_list.append([
                    clusters_list[j],
                    clusters_list[j + 1],
                    clusters_list[j + 2],
                    clusters_list[j + 3]
                ])

            for e in range(1, len(levels)):
                df_source.loc[mapped_idx, 'clusterL' + str(e)] = levels[e]

            if len(dimension_list) > 0:
                point = GHSOM_center_point(dimension_list)
                df_source.loc[mapped_idx, 'point_x'] = point[0]
                df_source.loc[mapped_idx, 'point_y'] = point[1]

        if str(saved_data_type) == 'result_detail':
            structure = {
                'name': unit_file_name,
                'group_position': group_position,
                'parent_name': parent_name,
                'sub_map_file_name': sub_map_file_name,
                'leaf_node_or_note': leaf_node,
                'statistic_info': current_group_statistic_info,
                'group_data_index': group_data_index,
                'cluster_string': cluster_string
            }

            if str(structure_type) == 'flat':
                Groups_info.append(structure)

    return Groups_info

def GHSOM_center_point(data_list):
    Bx = By = 1
    Bx_list = []
    By_list = []
    Point_list = []

    for i in range(len(data_list)):
        Bx *= Fraction(1, int(data_list[i][0]))
        Bx_list.append(Bx)

        By *= Fraction(1, int(data_list[i][1]))
        By_list.append(By)

        Point_list.append([Bx * int(data_list[i][2]), By * int(data_list[i][3])])

    Px = sum(p[0] for p in Point_list) + Bx_list[-1] * Fraction(1, 2)
    Py = sum(p[1] for p in Point_list) + By_list[-1] * Fraction(1, 2)

    return [Px, Py]

# ========================================
# Run clustering enrichment
# ========================================
saved_file_type = 'result_detail'
result = format_cluster_info_to_dict(prefix, df_source, saved_file_type, 'flat')

result_frame = pd.DataFrame(result)

# ========================================
# Save result
# ========================================
df_source.to_csv(
    f'./applications/{file}/data/{prefix}_with_clustered_label-{t1}-{t2}.csv',
    index=False
)








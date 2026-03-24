import pandas as pd
import csv
import numpy as np

def format_ghsom_input_vector(name, file, index, label, subnum, patient=None, time=None):
    """
    name : dataset name (string)
    file : application folder name (data-t1-t2)
    index : user-provided index column (string or None)
    label : user-provided label column (string or None)
    subnum : subsample number (int or None)
    patient : patient column (string or None) ⭐ NEW
    time : time column (string or None) ⭐ 建議一起補
    """

    print(subnum)

    # ============================
    # 讀 raw-data
    # ============================
    df = pd.read_csv(f'./raw-data/{name}.csv', encoding='utf-8')

    # ============================
    # 1️⃣ Drop metadata columns
    # ============================
    drop_cols = []

    if index is not None and index in df.columns:
        print(f"[INFO] Dropping index column: {index}")
        drop_cols.append(index)

    if label is not None and label in df.columns:
        print(f"[INFO] Dropping label column: {label}")
        drop_cols.append(label)

    if patient is not None and patient in df.columns:
        print(f"[INFO] Dropping patient column: {patient}")
        drop_cols.append(patient)

    if time is not None and time in df.columns:
        print(f"[INFO] Dropping time column: {time}")
        drop_cols.append(time)

    df = df.drop(columns=drop_cols, errors='ignore')

    # ============================
    # 2️⃣ Keep numeric columns only
    # ============================
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    # ============================
    # 3️⃣ Fill NA → 0
    # ============================
    df = df.fillna(0)

    # ============================
    # 4️⃣ Subsample
    # ============================
    if subnum is not None:
        df = df.sample(n=subnum)

    # ============================
    # printing info
    # ============================
    rows_amount = df.shape[0]
    columns_amount = df.shape[1]

    # ⚠️ 這裡避免 index=None 會炸
    if index is not None:
        df[index] = range(0, rows_amount)

    print('rows=', rows_amount)
    print('columns=', columns_amount)

    # ============================
    # 寫出 GHSOM input CSV
    # ============================
    ghsom_csv_path = f'./applications/{file}/GHSOM/data/{name}_ghsom.csv'
    df.to_csv(ghsom_csv_path, sep=' ', index=False, header=False)

    # ============================
    # 寫出 .in 檔
    # ============================
    ghsom_in_path = f'./applications/{file}/GHSOM/data/{name}_ghsom.in'

    data_type = 'inputvec'
    x_dim = rows_amount
    y_dim = 1
    vec_dim = columns_amount

    with open(ghsom_in_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([f'$TYPE {data_type}'])
        writer.writerow([f'$XDIM {x_dim}'])
        writer.writerow([f'$YDIM {y_dim}'])
        writer.writerow([f'$VECDIM {vec_dim}'])

        with open(ghsom_csv_path, 'r', newline='', encoding='utf-8') as rawfile:
            rows = csv.reader(rawfile)
            writer.writerow([])

            for row in rows:
                writer.writerow(row)

    print("[OK] GHSOM input formatting completed (final version).")












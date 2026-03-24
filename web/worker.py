import os
import sys
import time
import json

# ----------------------------------------------------------
# 專案根目錄（scGHSOM）
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ⭐ 強制切換到專案根目錄
os.chdir(BASE_DIR)

# ⭐ 讓 Python 找到 execute.py
sys.path.append(BASE_DIR)

from execute import run_pipeline

# ----------------------------------------------------------
# 資料夾路徑
# ----------------------------------------------------------
QUEUE_DIR = os.path.join(BASE_DIR, "web", "queue")
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw-data")
APPLICATION_DIR = os.path.join(BASE_DIR, "applications")
LABEL_BACKUP_DIR = os.path.join(BASE_DIR, "label")   # ←⭐ 你要的最外層資料夾

os.makedirs(LABEL_BACKUP_DIR, exist_ok=True)

print(f"[WORKER STARTED]")
print(f"Current working directory: {os.getcwd()}")
print(f"Queue directory: {QUEUE_DIR}")
print(f"Raw-data directory: {RAW_DATA_DIR}")
print(f"Label backup directory: {LABEL_BACKUP_DIR}")
print("========================================================")


# ----------------------------------------------------------
# Worker 無限循環
# ----------------------------------------------------------
while True:
    files = [f for f in os.listdir(QUEUE_DIR) if f.endswith(".json")]

    if not files:
        time.sleep(2)
        continue

    job_file = sorted(files)[0]
    job_path = os.path.join(QUEUE_DIR, job_file)

    print(f"\n[JOB FOUND] {job_file}")

    with open(job_path, "r") as f:
        job_info = json.load(f)

    job_id = job_info["job_id"]
    tau1 = job_info["tau1"]
    tau2 = job_info["tau2"]
    index = job_info.get("index")
    label = job_info.get("label")  # ←⭐ 使用者在前端填的 label 欄位名（可能為 None）

    print(f"[RUNNING JOB] job_id={job_id}")
    print(f"  tau1={tau1}, tau2={tau2}, index={index}, label={label}")

    try:
        run_pipeline(
            data=job_id,
            tau1=tau1,
            tau2=tau2,
            index=index,
            label=label
        )
        print(f"[JOB COMPLETED] {job_id}")

    except Exception as e:
        print(f"[ERROR] Job {job_id} failed: {e}")

    else:
        # ------------------------------------------------------
        # ⭐ Step 1：備份 label 欄位（若使用者有填）
        # ------------------------------------------------------
        raw_file = os.path.join(RAW_DATA_DIR, f"{job_id}.csv")

        if label is not None:
            try:
                import pandas as pd

                if os.path.exists(raw_file):
                    df_raw = pd.read_csv(raw_file)

                    if label in df_raw.columns:
                        df_label = df_raw[[label]]
                        backup_path = os.path.join(LABEL_BACKUP_DIR, f"{job_id}_label.csv")
                        df_label.to_csv(backup_path, index=False)
                        print(f"[LABEL SAVED] → {backup_path}")
                    else:
                        print(f"[WARNING] Label column '{label}' not found in raw CSV. Skip backup.")

                else:
                    print(f"[WARNING] Raw-data file missing, cannot backup label.")

            except Exception as e:
                print(f"[ERROR] Failed to backup label column: {e}")

        else:
            print(f"[INFO] User did not provide label. No label backup needed.")

        # ------------------------------------------------------
        # ⭐ Step 2：刪 raw-data CSV
        # ------------------------------------------------------
        raw_file_path = os.path.join(RAW_DATA_DIR, f"{job_id}.csv")
        if os.path.exists(raw_file_path):
            os.remove(raw_file_path)
            print(f"[RAW DATA CLEANED] Removed {job_id}.csv")
        else:
            print(f"[RAW DATA MISSING] {raw_file_path} not found")
        
        # ------------------------------------------------------
        # ⭐ Step 2.5：寄信通知使用者（如果 gmail 有填）
        # ------------------------------------------------------
        user_email = job_info.get("gmail")

        if user_email:
            try:
                from web.mailer import send_done_mail
                send_done_mail(user_email, job_id)
            except Exception as e:
                print(f"[EMAIL ERROR] Failed to send mail: {e}")
        else:
            print("[INFO] No gmail provided. Skip sending mail.")


        # ------------------------------------------------------
        # ⭐ Step 3：刪 queue JSON（最後才刪）
        # ------------------------------------------------------
        os.remove(job_path)
        print(f"[QUEUE CLEANED] Removed {job_file}")

    print("--------------------------------------------------------")
    time.sleep(1)



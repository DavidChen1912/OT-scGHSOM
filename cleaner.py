# 每七天負責定情清理 applications job_meta result label四個資料夾，example檔案永久不刪除
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_DIRS = [
    os.path.join(BASE_DIR, "applications"),
    os.path.join(BASE_DIR, "Result"),
    os.path.join(BASE_DIR, "web", "job_meta"),
    os.path.join(BASE_DIR, "label")
]

def should_delete(name: str) -> bool:
    """
    Return True if this file/folder should be deleted.
    DO NOT delete anything whose name contains 'example'.
    """
    return "example" not in name.lower()

def clean_folder(path: str):
    if not os.path.exists(path):
        print(f"[SKIP] {path} does not exist.")
        return

    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if should_delete(item):
            # Delete file
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"[DELETE FILE] {item_path}")

            # Delete folder
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"[DELETE FOLDER] {item_path}")

        else:
            print(f"[KEEP] {item_path}")

def main():
    print("========== CLEANING START ==========")
    for folder in TARGET_DIRS:
        print(f"\n[Folder] {folder}")
        clean_folder(folder)
    print("=========== CLEANING DONE ==========")

if __name__ == "__main__":
    main()

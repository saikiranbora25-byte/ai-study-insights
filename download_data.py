# download_data.py
# -----------------------------------------------------------------------------
# Helper script to download the Kaggle dataset.
# Prerequisites:
#   1. pip install kaggle
#   2. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)
# -----------------------------------------------------------------------------

import os
import shutil
import zipfile

DATASET   = "imtkaggleteam/how-ai-changing-life-of-students"   # update if slug differs
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading dataset: {DATASET} ...")
os.system(f"kaggle datasets download -d {DATASET} --path {OUTPUT_DIR} --unzip")

# Rename any CSV found to our expected name
for fname in os.listdir(OUTPUT_DIR):
    if fname.endswith(".csv") and fname != "students_ai_india.csv":
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(OUTPUT_DIR, "students_ai_india.csv")
        shutil.move(src, dst)
        print(f"Renamed {fname} -> students_ai_india.csv")
        break

print("Done! File saved to data/students_ai_india.csv")
print("\nYou can now run the app:")
print("  streamlit run app.py")

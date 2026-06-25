import os
import sys
import shutil
import json
import papermill as pm
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi

# ============================================================
# PATHS
# ============================================================
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = os.path.join(WORKSPACE_ROOT, "src")
NOTEBOOKS_DIR = os.path.join(WORKSPACE_ROOT, "notebooks")
SCRIPTS_DIR = os.path.join(WORKSPACE_ROOT, "scripts")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")
BUILD_DIR = os.path.join(WORKSPACE_ROOT, "kaggle_build")

# Make sure folders exist
os.makedirs(BUILD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
GPU_TYPE = "T4"  # "T4", "P100", "V100", or None
KERNEL_USERNAME = "maran"  # CHANGE to your username

TEMPLATE_NOTEBOOK = os.path.join(NOTEBOOKS_DIR, "siamese_template.ipynb")
MAIN_NOTEBOOK = os.path.join(NOTEBOOKS_DIR, "siamese.ipynb")
RUN_NOTEBOOK = os.path.join(NOTEBOOKS_DIR, "siamese_template_run.ipynb")

PARAMS = {
    "target_col": "bfrb",
    "search_mode": "bayesian",
    "n_iter": 50,
    "use_eg_sample": False,
}

# Files to copy to Kaggle
FILES_TO_COPY = [
    "siamese.ipynb",
    "base_utils_qwen.py",
    "data_utils.py",
    "utils_siamese_contrastive.py",
    "multi_rocket_utils.py",
]

# Unique title
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
KERNEL_TITLE = f"siamese_run_{timestamp}"

print(f"Kernel Title: {KERNEL_TITLE}")

# ============================================================
# 1. GENERATE NOTEBOOK WITH PARAMETERS
# ============================================================
pm.execute_notebook(
    TEMPLATE_NOTEBOOK,
    RUN_NOTEBOOK,
    parameters=PARAMS,
    progress_bar=False
)
print(f"✅ Generated: {RUN_NOTEBOOK}")

# ============================================================
# 2. PREPARE KAGGLE FOLDER
# ============================================================
os.makedirs(BUILD_DIR, exist_ok=True)

# Copy the generated notebook
shutil.copy(RUN_NOTEBOOK, os.path.join(BUILD_DIR, "run.ipynb"))

# Copy all required files
for file in FILES_TO_COPY:
    # Check both src/ and notebooks/ directories
    src_paths = [
        os.path.join(SRC_DIR, file),
        os.path.join(NOTEBOOKS_DIR, file),
    ]
    found = False
    for src_path in src_paths:
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(BUILD_DIR, file))
            print(f"✅ Copied: {file}")
            found = True
            break
    if not found:
        print(f"⚠️ Missing: {file}")

# ============================================================
# 3. CREATE METADATA
# ============================================================
metadata = {
    "id": f"{KERNEL_USERNAME}/{KERNEL_TITLE}",
    "title": KERNEL_TITLE,
    "code_file": "run.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": GPU_TYPE is not None,
    "enable_internet": False,
    "dataset_sources": [],
    "competition_sources": ["cmi-detect-behavior-with-sensor-data"],
}

if GPU_TYPE in ["T4", "P100", "V100"]:
    metadata["gpu_type"] = GPU_TYPE
    print(f"✅ GPU selected: {GPU_TYPE}")

with open(os.path.join(BUILD_DIR, "kernel-metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ============================================================
# 4. PUSH TO KAGGLE
# ============================================================
api = KaggleApi()
api.authenticate()
api.kernels_push(BUILD_DIR)

print(f"✅ Kernel '{KERNEL_TITLE}' pushed to Kaggle!")
print(f"   View at: https://www.kaggle.com/{KERNEL_USERNAME}/{KERNEL_TITLE}")
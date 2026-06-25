import os
from kaggle.api.kaggle_api_extended import KaggleApi

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")

api = KaggleApi()
api.authenticate()

KERNEL_TITLE = input("Enter kernel title: ")  # or hardcode

# Download
api.kernels_output(KERNEL_TITLE, path=RESULTS_DIR)
print(f"✅ Results downloaded to {RESULTS_DIR}")
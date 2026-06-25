import os
import time
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

KERNEL_TITLE = input("Enter kernel title: ")

while True:
    status = api.kernels_status(KERNEL_TITLE)
    print(f"Status: {status['status']}")
    if status['status'] in ['complete', 'error']:
        break
    time.sleep(30)

# Get logs
logs = api.kernels_logs(KERNEL_TITLE, text=True)
print(logs)
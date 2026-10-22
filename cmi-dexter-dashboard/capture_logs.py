"""Capture official Kaggle live log events; the caller bounds the process lifetime."""
import json
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    for event in api.kernels_logs_stream(sys.argv[1]):
        print(json.dumps(event), flush=True)


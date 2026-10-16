"""Profile preprocessing on a deterministic wide sensor table, without model training.

Use --source PATH --separate-fit-transform to compare an older base utility file.
Peak memory counts traced preprocessing allocations, excluding the resident input.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "src" / "base_utils.py")
    parser.add_argument("--separate-fit-transform", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_profiled_base_utils", args.source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rng = np.random.default_rng(42)
    n_sequences, length = 48, 96
    n = n_sequences * length
    values = {
        "sequence_id": np.repeat(np.arange(n_sequences), length),
        "sequence_counter": np.tile(np.arange(length), n_sequences),
    }
    for name in ("acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"):
        values[name] = rng.normal(size=n)
    for sensor in range(1, 6):
        values[f"thm_{sensor}"] = rng.normal(size=n)
        for pixel in range(64):
            values[f"tof_{sensor}_v{pixel}"] = rng.uniform(0, 255, size=n)
    frame = pd.DataFrame(values)
    extractor = module.SequenceExtractor(
        acc_modes="raw", rotation_modes="quaternion", tof_modes="raw", thm_modes="raw",
        stft_configs=None, cwt_configs=None, add_global_context=True,
        chunk_window_size=48, chunk_stride=24, maxlen=96,
    )
    tracemalloc.start()
    start = time.perf_counter()
    if args.separate_fit_transform:
        extractor.fit(frame)
        chunks = extractor.transform_chunks(frame)
    else:
        chunks = extractor.fit_transform(frame)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    digest = hashlib.sha256()
    for key in ("X", "mask", "sequence_ids"):
        digest.update(chunks[key].tobytes())
    digest.update(json.dumps(chunks["feature_names"]).encode())
    print(json.dumps({
        "source": str(args.source), "input_shape": frame.shape,
        "input_mib": frame.memory_usage(deep=True).sum() / 1024**2,
        "output_shape": chunks["X"].shape, "elapsed_seconds": elapsed,
        "peak_traced_mib": peak / 1024**2, "sha256": digest.hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()

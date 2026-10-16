"""Exercise the notebook's actual orientation eligibility policy before splitting."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def notebook_orientation_filter():
    notebook = json.loads((ROOT / "notebooks" / "multibranch_v1.ipynb").read_text(encoding="utf-8"))
    cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "quaternion_cols =" in "".join(cell["source"])
    )
    statements = ast.parse(cell).body
    start = next(
        index for index, node in enumerate(statements)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "quaternion_cols" for target in node.targets)
    )
    end = next(index for index in range(start, len(statements)) if isinstance(statements[index], ast.With))
    block = ast.Module(body=statements[start:end + 1], type_ignores=[])
    return compile(block, str(ROOT / "notebooks" / "multibranch_v1.ipynb"), "exec")


class NotebookOrientationEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filter_code = notebook_orientation_filter()

    def run_filter(self, frame):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            namespace = {
                "train_df": frame,
                "np": np,
                "pd": pd,
                "json": json,
                "results_dir": results_dir,
                "timestamp": "regression",
            }
            with redirect_stdout(StringIO()):
                exec(self.filter_code, namespace)
            manifest = json.loads((results_dir / "regression_input_exclusions.json").read_text())
            return namespace["train_df"], manifest

    def test_missing_sequence_is_excluded_without_modifying_retained_samples(self):
        # Interleaved sequences and noncontiguous indices catch accidental row-level
        # filtering, index misalignment, and interpolation across sequence boundaries.
        frame = pd.DataFrame({
            "sequence_id": ["valid", "missing", "partial", "valid", "partial", "missing"],
            "sequence_counter": [0, 0, 0, 1, 1, 1],
            "rot_w": [1.0, np.nan, np.nan, 0.5, 1.0, np.nan],
            "rot_x": [0.0, np.nan, np.nan, 0.5, 0.0, np.nan],
            "rot_y": [0.0, np.nan, np.nan, 0.5, 0.0, np.nan],
            "rot_z": [0.0, np.nan, np.nan, 0.5, 0.0, np.nan],
            "acc_x": [3.0, 9.0, 7.0, 4.0, 8.0, 10.0],
            "gesture": ["one", "two", "three", "one", "three", "two"],
        }, index=[10, 20, 40, 70, 80, 100])
        original = frame.copy(deep=True)

        retained, manifest = self.run_filter(frame)

        pd.testing.assert_frame_equal(retained, original.loc[[10, 40, 70, 80]])
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(manifest, {
            "reason": "no finite nonzero quaternion in sequence",
            "sequence_ids": ["missing"],
            "evaluation_scope": "sequences with orientation data",
        })

    def test_fully_observed_input_is_unchanged_and_records_no_exclusions(self):
        frame = pd.DataFrame({
            "sequence_id": ["a", "a", "b"],
            "rot_w": [1.0, -1.0, 0.0],
            "rot_x": [0.0, 0.0, 1.0],
            "rot_y": [0.0, 0.0, 0.0],
            "rot_z": [0.0, 0.0, 0.0],
        })

        retained, manifest = self.run_filter(frame)

        pd.testing.assert_frame_equal(retained, frame)
        self.assertEqual(manifest["sequence_ids"], [])

    def test_zero_and_nonfinite_quaternions_do_not_count_as_observed_orientation(self):
        frame = pd.DataFrame({
            "sequence_id": ["zero", "nonfinite", "partial", "partial"],
            "rot_w": [0.0, np.inf, np.nan, 1.0],
            "rot_x": [0.0, 0.0, np.nan, 0.0],
            "rot_y": [0.0, 0.0, np.nan, 0.0],
            "rot_z": [0.0, 0.0, np.nan, 0.0],
        })

        retained, manifest = self.run_filter(frame)

        pd.testing.assert_frame_equal(retained, frame.iloc[2:])
        self.assertCountEqual(manifest["sequence_ids"], ["zero", "nonfinite"])


if __name__ == "__main__":
    unittest.main()

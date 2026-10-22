"""Execute notebook reporting against the runner's real result schema, without fitting."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pretraining_search import run_validation_searches


class PretrainingReportingSchemaTests(unittest.TestCase):
    def test_reporting_cell_accepts_actual_runner_result_without_training(self):
        notebook = json.loads(
            (ROOT / "notebooks/multibranch_pretraining.ipynb").read_text(encoding="utf-8")
        )
        reporting_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "Experiments CSV:" in "".join(cell["source"])
        ]
        self.assertEqual(len(reporting_cells), 1)

        with TemporaryDirectory() as directory:
            # Missing subject metadata stops before cloning or fitting any search.
            # The real runner still saves both artifacts and returns its actual schema.
            with redirect_stdout(StringIO()):
                result = run_validation_searches(
                    object(), pd.DataFrame({"sequence_id": ["s1"]}),
                    validation_param_space={"validation_mode": ["subject"]},
                    sensor_columns=[], train_size=0.7, n_splits=1,
                    random_state=42, target_col="bfrb", results_dir=directory,
                    experiment_name="reporting", timestamp="run",
                )
            self.assertEqual(result["protocol_results"][0]["status"], "failed")
            self.assertIn("Missing group column", result["protocol_results"][0]["error"])
            self.assertTrue(result["experiments_path"].is_file())
            self.assertTrue(result["summary_path"].is_file())

            output = StringIO()
            with redirect_stdout(output):
                exec(compile(reporting_cells[0], "notebook-reporting", "exec"),
                     {"validation_runs": result})
            self.assertEqual(output.getvalue().splitlines(), [
                f"Experiments CSV: {result['experiments_path']}",
                f"Summary JSON: {result['summary_path']}",
            ])


if __name__ == "__main__":
    unittest.main()

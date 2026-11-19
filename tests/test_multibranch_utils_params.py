"""Shared pipeline helpers for multibranch_utils hierarchical search unit tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import multibranch_utils as mba  # noqa: E402
from base_utils import (  # noqa: E402
    ECat,
    EFloat,
    EInt,
    GASearchCV,
    SensorAugmentor,
    SequenceExtractor,
)

NOTEBOOK = ROOT / "notebooks" / "multibranch_alpha.ipynb"


def make_pipeline():
    return Pipeline(
        [
            (
                "estimator",
                mba.MultiBranchSequenceClassifier(
                    extractor=SequenceExtractor(output_format="chunks"),
                    augmentor=SensorAugmentor(
                        sequence_col="sequence_id",
                        counter_col="sequence_counter",
                        prob=0.0,
                        per_aug_prob=0.0,
                    ),
                    primary_target="bfrb",
                ),
            ),
        ]
    )


class MultibranchUtilsPipelineSanityTests(unittest.TestCase):
    def test_make_pipeline_exposes_estimator_params(self):
        params = make_pipeline().get_params(deep=True)
        self.assertIn("estimator__temporal_architecture", params)
        self.assertIn("estimator__primary_target", params)


if __name__ == "__main__":
    unittest.main()

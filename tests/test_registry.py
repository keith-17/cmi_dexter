"""Unit tests for FEATURE_REGISTRY extensibility, worked plugin template, and FeatureManifest."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import base_utils as bu
from base_utils import (
    ALGO_VERSION,
    BUILTIN_WINDOW_CELLS,
    FEATURE_REGISTRY,
    FEATURE_VERSION,
    BaseAugmentor,
    BaseSearchCV,
    BaseSequenceClassifier,
    BaseSequenceExtractor,
    ExampleMeanCell,
    FeatureBackend,
    FeatureManifest,
    HoneycombBase,
    WindowFeatureCell,
)
from utils import (
    HierarchicalBayesSearchCV,
    MultiBranchGASearchCV,
    MultiBranchTSAIClassifier,
    MultiModalSequenceExtractor,
    NoopOptions,
    TSAugmentor,
    make_synthetic_cmi_df,
)


class RegistryAndManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_synthetic_cmi_df(n_subjects=2, sequences_per_subject=2, n_timesteps=50, random_state=13)

    def tearDown(self):
        ExampleMeanCell.register(FEATURE_REGISTRY)

    def test_honeycomb_parent_child_class_hierarchy(self):
        self.assertTrue(issubclass(bu.InvalidExtractorParams, bu.InvalidParams))
        self.assertTrue(issubclass(bu.ProblematicSequenceFilter, HoneycombBase))
        self.assertTrue(issubclass(bu.HandednessCanonicalizer, HoneycombBase))
        self.assertTrue(issubclass(bu.DenoisingFilterCell, HoneycombBase))
        self.assertTrue(issubclass(WindowFeatureCell, HoneycombBase))
        for cell_cls in BUILTIN_WINDOW_CELLS:
            self.assertTrue(issubclass(cell_cls, WindowFeatureCell), f"{cell_cls} must inherit WindowFeatureCell")
        self.assertTrue(issubclass(bu.SequenceExtractor, BaseSequenceExtractor))
        self.assertTrue(issubclass(MultiModalSequenceExtractor, BaseSequenceExtractor))
        self.assertTrue(issubclass(MultiModalSequenceExtractor, HoneycombBase))
        self.assertTrue(issubclass(bu.SensorAugmentor, BaseAugmentor))
        self.assertTrue(issubclass(TSAugmentor, BaseAugmentor))
        self.assertTrue(issubclass(bu.RandomForestSequenceClassifier, BaseSequenceClassifier))
        self.assertTrue(issubclass(MultiBranchTSAIClassifier, BaseSequenceClassifier))
        self.assertTrue(issubclass(bu.GASearchCV, BaseSearchCV))
        self.assertTrue(issubclass(MultiBranchGASearchCV, bu.GASearchCV))
        self.assertTrue(issubclass(HierarchicalBayesSearchCV, BaseSearchCV))
        for name in FEATURE_REGISTRY.list_backends():
            self.assertIsInstance(FEATURE_REGISTRY.get_backend(name), FeatureBackend)

    def test_plugin_template_runs_through_both_extractors_without_core_edits(self):
        FEATURE_REGISTRY.unregister("example_mean")
        spec = ExampleMeanCell.register(FEATURE_REGISTRY)
        self.assertIn(spec.name, FEATURE_REGISTRY.list_families())
        self.assertIn(spec.name, FEATURE_REGISTRY.list_extensions())

        # 1. Through utils.MultiModalSequenceExtractor
        ext_v3 = MultiModalSequenceExtractor(
            window_seconds=0.5,
            static_backend="none",
            extra_feature_families="example_mean",
        )
        chunks = ext_v3.fit(self.df).transform_chunks(self.df)
        self.assertIn("example_mean__acc_mag", chunks["static_names"])
        self.assertTrue(np.isfinite(chunks["static"]).all())

        # 2. Through base_utils.SequenceExtractor
        ext_base = bu.SequenceExtractor(
            acc_modes="raw",
            use_acc_magnitude=True,
            rotation_modes=None,
            tof_modes=None,
            thm_modes=None,
            stft_configs=None,
            cwt_configs=None,
            output_format="frame",
            extra_feature_families="example_mean",
        )
        out_df = ext_base.fit_transform(self.df)
        matched = [c for c in out_df.columns if c.startswith("example_mean__acc_mag")]
        self.assertGreater(len(matched), 0, f"Plugin column missing from base_utils output: {list(out_df.columns)}")
        self.assertTrue(np.isfinite(out_df[matched].to_numpy()).all())

    def test_feature_manifest_json_and_provenance(self):
        ext = MultiModalSequenceExtractor(
            window_seconds=0.5,
            static_backend="catch22",
            static_channels="acc_mag",
            noop=NoopOptions(motion_stats="basic"),
            orientation_features=True,
            thermal_features=True,
        )
        chunks = ext.fit(self.df).transform_chunks(self.df)
        manifest = ext.feature_manifest_
        self.assertIsInstance(manifest, FeatureManifest)
        self.assertIs(chunks["feature_manifest"], manifest)

        payload = manifest.to_dict()
        self.assertEqual(payload["algo_version"], ALGO_VERSION)
        self.assertEqual(payload["feature_version"], FEATURE_VERSION)
        self.assertIn("orientation_posture", payload["families"])
        self.assertIn("thermal_dynamics", payload["families"])
        self.assertIn("signal_descriptors", payload["families"])
        self.assertIn("external_catch22", payload["families"])

        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "manifest.json"
            raw_json = manifest.to_json(out_path)
            parsed = json.loads(raw_json)
            self.assertEqual(parsed["feature_count"], len(manifest.entries))
            self.assertTrue(out_path.is_file())


if __name__ == "__main__":
    unittest.main()

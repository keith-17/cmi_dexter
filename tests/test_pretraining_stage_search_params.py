"""Editable stage choices survive notebook assembly and reach real selectors."""
import ast
from contextlib import redirect_stdout
from io import StringIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone

import test_pretraining_large_notebook as large
from test_pretraining_sequence_selection import recordings


PREFIX = "estimator__"
EXTRACTOR = PREFIX + "extractor__"
STAGE_DEFAULTS = {
    PREFIX + "pretraining_epochs": [30],
    PREFIX + "epochs": [150],
    PREFIX + "learning_rate": [1e-4],
    PREFIX + "pretraining_learning_rate": [1e-4],
    PREFIX + "finetuning_learning_rate": [None],
    EXTRACTOR + "pretraining_source": ["non_target", "all", "target", "baseline"],
    EXTRACTOR + "pretraining_label_column": ["gesture"],
    EXTRACTOR + "pretraining_include_classes": [None],
    EXTRACTOR + "pretraining_exclude_classes": [None],
    EXTRACTOR + "pretraining_filters": [None],
    EXTRACTOR + "pretraining_exclude_filters": [None],
    EXTRACTOR + "pretraining_min_length": [None],
    EXTRACTOR + "pretraining_max_length": [None],
    EXTRACTOR + "pretraining_sample_fraction": [1.0],
    EXTRACTOR + "pretraining_max_sequences": [None],
    EXTRACTOR + "pretraining_max_sequences_per_class": [None],
    EXTRACTOR + "pretraining_balance_classes": [False],
    EXTRACTOR + "finetuning_source": ["target"],
    EXTRACTOR + "finetuning_include_classes": [None],
    EXTRACTOR + "finetuning_exclude_classes": [None],
    EXTRACTOR + "finetuning_filters": [None],
    EXTRACTOR + "finetuning_exclude_filters": [None],
    EXTRACTOR + "stage_target_column": ["is_target"],
    EXTRACTOR + "stage_target_values": [(1,)],
}


def insert_after_assignment(source, name, extra):
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == name
                           for target in node.targets)]
    if len(assignments) != 1:
        raise AssertionError(f"Expected one editable {name} assignment")
    lines = source.splitlines(keepends=True)
    lines.insert(assignments[0].end_lineno, "\n" + extra + "\n")
    return "".join(lines)


def decoded(value):
    value = large.mb._json_maybe(value)
    return list(value) if isinstance(value, tuple) else value


class StageSearchParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        large.LargePretrainingNotebookTests.setUpClass()
        cls.harness = large.LargePretrainingNotebookTests(methodName="runTest")

    def namespace(self, variant, directory, edits=None):
        return self.harness.namespace(variant, directory, source_edits=edits)

    def assert_compiled_choices(self, namespace, key, expected):
        self.assertEqual(namespace["GRID_PARAM_SPACE"][key], expected)
        if namespace["SKOPT_AVAILABLE"]:
            bayesian = namespace["BAYESIAN_PARAM_SPACE"][key]
            self.assertEqual([decoded(value) for value in bayesian.categories],
                             [decoded(value) for value in expected], key)
        evolutionary = namespace["EVOLUTIONARY_SEARCH_SPACE"][key]
        self.assertEqual([decoded(value) for value in evolutionary.choices],
                         [decoded(value) for value in expected], key)

    def test_all_requested_stage_controls_are_editable_in_both_searches(self):
        for variant in ("model1", "model2"):
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory)
                stage = namespace["PRETRAINING_PARAM_SPACE"]
                for key, choices in STAGE_DEFAULTS.items():
                    self.assertIn(key, stage)
                    self.assertEqual(stage[key], choices)
                    self.assert_compiled_choices(namespace, key, choices)
                actual = clone(namespace["pipeline"])
                for key, choices in STAGE_DEFAULTS.items():
                    actual.set_params(**{key: choices[-1]})
                    self.assertEqual(actual.get_params(deep=True)[key], choices[-1])

    def test_expanded_choices_survive_late_large_run_overrides(self):
        choices = {
            PREFIX + "pretraining_epochs": [3, 11],
            PREFIX + "epochs": [7, 21],
            PREFIX + "learning_rate": [2e-5, 3e-4],
            PREFIX + "pretraining_learning_rate": [1e-5, 2e-4],
            EXTRACTOR + "pretraining_include_classes": [None, ["wave", "glasses"]],
            EXTRACTOR + "pretraining_filters": [None, {"subject": ["p2"]}],
            EXTRACTOR + "pretraining_sample_fraction": [0.5, 1.0],
            EXTRACTOR + "stage_target_values": [(1,), (0,)],
        }

        def edits(index, source):
            if index == 14:
                return insert_after_assignment(source, "PRETRAINING_PARAM_SPACE",
                                               "PRETRAINING_PARAM_SPACE.update(" + repr(choices) + ")")
            return source

        for variant in ("model1", "model2"):
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory, edits)
                for key, values in choices.items():
                    self.assert_compiled_choices(namespace, key, values)
                for space_name in ("BAYESIAN_PARAM_SPACE", "EVOLUTIONARY_SEARCH_SPACE"):
                    candidate = {key: decoded(getattr(namespace[space_name][key],
                                                     "categories", getattr(namespace[space_name][key],
                                                                           "choices", []))[-1])
                                 for key in choices}
                    copy = clone(namespace["pipeline"]).set_params(**candidate)
                    self.assertEqual(copy.named_steps["estimator"].epochs, 21)
                    self.assertEqual(copy.named_steps["estimator"].pretraining_epochs, 11)
                    extractor = copy.named_steps["estimator"].extractor
                    self.assertEqual(decoded(extractor.pretraining_filters), {"subject": ["p2"]})

    def test_searched_class_metadata_length_and_sampling_choices_select_whole_sequences(self):
        frame = recordings()
        candidates = (
            ({"pretraining_source": "all", "pretraining_include_classes": ["wave", "scratch"],
              "pretraining_filters": {"subject": ["p1", "p2"]},
              "pretraining_exclude_filters": {"subject": ["p1"]},
              "pretraining_min_length": 5, "pretraining_max_length": 7}, {"c"}),
            ({"pretraining_source": "all", "pretraining_exclude_classes": ["wave"],
              "pretraining_exclude_filters": {"subject": ["p3"]}}, {"a", "d"}),
            ({"pretraining_source": "target", "stage_target_column": "is_target",
              "stage_target_values": (0,)}, {"b", "c", "d", "f"}),
        )
        with TemporaryDirectory() as directory:
            namespace = self.namespace("model1", directory)
            for values, expected in candidates:
                for compiler in (large.mb.prepare_multibranch_bayesian_space,
                                 large.mb.prepare_multibranch_evolutionary_space):
                    with self.subTest(candidate=values, compiler=compiler.__name__):
                        compiled = compiler({EXTRACTOR + key: [value] for key, value in values.items()})
                        selected = {key: getattr(value, "categories", getattr(value, "choices", []))[0]
                                    for key, value in compiled.items()}
                        extractor = clone(namespace["pipeline"]).set_params(**selected).named_steps["estimator"].extractor
                        result = extractor.select_pretraining_sequences(frame)
                        self.assertEqual(set(result.sequence_id), expected)
                        for sequence in expected:
                            self.assertEqual(len(result[result.sequence_id == sequence]),
                                             len(frame[frame.sequence_id == sequence]))
            estimator = clone(namespace["pipeline"]).named_steps["estimator"]
            estimator.set_params(extractor__pretraining_source="all",
                                 extractor__pretraining_sample_fraction=0.5,
                                 extractor__pretraining_max_sequences=2,
                                 extractor__pretraining_max_sequences_per_class=1,
                                 extractor__pretraining_balance_classes=True,
                                 extractor__pretraining_random_state=9)
            first = estimator.extractor.select_pretraining_sequences(frame)
            second = estimator.extractor.select_pretraining_sequences(frame)
            large.pd.testing.assert_frame_equal(first, second)
            self.assertLessEqual(first.sequence_id.nunique(), 2)
            self.assertTrue((first.drop_duplicates("sequence_id").gesture.value_counts() <= 1).all())

    def test_evolutionary_integer_float_ranges_survive_final_compilation(self):
        def edits(index, source):
            if index == 14:
                return insert_after_assignment(source, "EVOLUTIONARY_STAGE_OVERRIDES", """
EVOLUTIONARY_STAGE_OVERRIDES.update({
    'estimator__pretraining_epochs': EInt(3, 12),
    'estimator__learning_rate': EFloat(1e-5, 1e-3, distribution='log-uniform'),
    'estimator__extractor__pretraining_sample_fraction': EFloat(0.25, 1.0),
})
""")
            return source

        for variant in ("model1", "model2"):
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory, edits)
                for name in ("EVOLUTIONARY_PARAM_SPACE", "EVOLUTIONARY_SEARCH_SPACE"):
                    space = namespace[name]
                    self.assertEqual((space[PREFIX + "pretraining_epochs"].low,
                                      space[PREFIX + "pretraining_epochs"].high), (3, 12))
                    self.assertEqual((space[PREFIX + "learning_rate"].low,
                                      space[PREFIX + "learning_rate"].high), (1e-5, 1e-3))
                    self.assertEqual((space[EXTRACTOR + "pretraining_sample_fraction"].low,
                                      space[EXTRACTOR + "pretraining_sample_fraction"].high), (0.25, 1.0))
                self.assertEqual(namespace["GRID_PARAM_SPACE"][PREFIX + "pretraining_epochs"], [30])
                self.assertEqual(namespace["GRID_PARAM_SPACE"][PREFIX + "learning_rate"], [1e-4])

    def test_finetuning_domain_and_filters_reach_the_actual_selector(self):
        frame = recordings()
        params = {
            EXTRACTOR + "finetuning_source": ["all"],
            EXTRACTOR + "finetuning_label_column": ["gesture"],
            EXTRACTOR + "finetuning_include_classes": [["scratch", "wave", "glasses"]],
            EXTRACTOR + "finetuning_exclude_classes": [["glasses"]],
            EXTRACTOR + "finetuning_filters": [{"subject": ["p1", "p2"]}],
            EXTRACTOR + "finetuning_exclude_filters": [{"subject": ["p1"]}],
        }
        with TemporaryDirectory() as directory:
            namespace = self.namespace("model1", directory)
            for compiler in (large.mb.prepare_multibranch_bayesian_space,
                             large.mb.prepare_multibranch_evolutionary_space):
                compiled = compiler(params)
                candidate = {key: getattr(value, "categories", getattr(value, "choices", []))[0]
                             for key, value in compiled.items()}
                estimator = clone(namespace["pipeline"]).set_params(**candidate).named_steps["estimator"]
                selected = estimator.extractor.select_finetuning_sequences(frame)
                self.assertEqual(set(selected.sequence_id), {"c"})
                self.assertEqual(len(selected), len(frame[frame.sequence_id == "c"]))

    def test_first_run_budgets_only_replace_stage_epochs_when_enabled(self):
        def edits(index, source):
            if index == 3:
                source = source.replace("FIRST_RUN = False", "FIRST_RUN = True")
            elif index == 14:
                source = insert_after_assignment(source, "PRETRAINING_PARAM_SPACE", """
PRETRAINING_PARAM_SPACE['estimator__pretraining_epochs'] = [30, 45]
PRETRAINING_PARAM_SPACE['estimator__epochs'] = [150, 200]
""")
            return source

        for variant in ("model1", "model2"):
            with self.subTest(variant=variant), TemporaryDirectory() as directory:
                namespace = self.namespace(variant, directory, edits)
                pilot = namespace["FIRST_RUN_PARAM_SPACE"]
                self.assertEqual(pilot[PREFIX + "pretraining_epochs"], [5])
                self.assertEqual(pilot[PREFIX + "epochs"], [5])
                self.assertTrue(namespace["FIRST_RUN"])
                self.assert_compiled_choices(namespace, PREFIX + "pretraining_epochs", [5])
                self.assert_compiled_choices(namespace, PREFIX + "epochs", [5])
                estimator = namespace["pipeline"].named_steps["estimator"]
                self.assertEqual((estimator.pretraining_epochs, estimator.epochs), (5, 5))
                self.assertEqual(namespace["PRETRAINING_PARAM_SPACE"][PREFIX + "epochs"], [150, 200])

    def test_searched_epochs_and_separate_learning_rates_reach_training_calls(self):
        frame = recordings()
        captured = {}
        with TemporaryDirectory() as directory:
            namespace = self.namespace("model1", directory)
            estimator = clone(namespace["pipeline"]).named_steps["estimator"]
            estimator.set_params(pretraining_epochs=3, epochs=7, pretraining_batch_size=2,
                                 learning_rate=2e-4, pretraining_learning_rate=3e-4,
                                 finetuning_learning_rate=None, pretraining_class_weight_mode=None)
            estimator.extractor_ = clone(estimator.extractor)
            estimator._pretraining_X = frame
            estimator._pretraining_y = frame
            estimator._pretraining_validation_X = None
            estimator.branch_groups_ = {}
            ids = frame.sequence_id.drop_duplicates().to_numpy()
            chunks = {"X": np.zeros((len(ids), 2, 1)), "mask": np.ones((len(ids), 2)),
                      "sequence_ids": ids}

            def fit(*args, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(history={"loss": [1.0]})

            source_model = SimpleNamespace(layers=[], fit=fit)
            with patch.object(estimator.extractor_, "transform_chunks", return_value=chunks), \
                    patch.object(estimator, "_build_model", return_value=source_model) as build, \
                    patch.object(estimator, "_split_branches", return_value=[]), \
                    patch.object(estimator, "_model_inputs", return_value=[]), \
                    patch.object(estimator, "_pretraining_callbacks", return_value=[]):
                estimator._supervised_pretrain({}, {}, 2, estimator._pretraining_settings())
            self.assertEqual(captured["epochs"], 3)
            self.assertEqual(captured["batch_size"], 2)
            self.assertEqual(build.call_args.kwargs["learning_rate"], 3e-4)
            self.assertEqual(estimator.epochs, 7)
            assigned = []
            estimator.pretraining_baseline_ = True
            estimator.model_ = SimpleNamespace(optimizer=SimpleNamespace(
                learning_rate=SimpleNamespace(assign=assigned.append)))
            estimator._before_supervised_fit(time_series_shapes={}, static_shapes={}, sequence_length=2,
                                             model_inputs=[], train_targets=[], sample_weight=None,
                                             validation_data=None)
            self.assertEqual(assigned, [2e-4])


if __name__ == "__main__":
    unittest.main()

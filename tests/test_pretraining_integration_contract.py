"""Cross-module contracts exercised without expensive TensorFlow fitting."""
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import base_utils as data_utils
import multibranch_utils as mb
import pretraining_utils as pretraining


def recordings():
    rows = []
    for wearer in range(20):
        for target in (0, 1):
            for gesture in range(2):
                for time in range(3):
                    rows.append({
                        'subject': f's{wearer}', 'sequence_id': f's{wearer}_{target}_{gesture}',
                        'is_target': target, 'gesture': f'g{target}_{gesture}',
                        'bfrb': f'g{target}_{gesture}' if target else 'non_bfrb',
                        'sequence_counter': time, 'acc_x': float(time),
                    })
    frame = pd.DataFrame(rows)
    return frame, frame[['subject', 'sequence_id', 'is_target', 'gesture', 'bfrb']].copy()


class PretrainingIntegrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((ROOT / 'archive/notebooks/multibranch_pretraining_v1.ipynb').read_text(encoding='utf-8'))

    def source(self, index):
        return ''.join(self.notebook['cells'][index]['source'])

    def namespace(self, directory):
        namespace = {
            'workspace_root': directory, 'Path': Path, 'datetime': datetime,
            'np': np, 'pd': pd, 'json': json, 'GroupShuffleSplit': GroupShuffleSplit,
            'GroupKFold': GroupKFold, 'Pipeline': Pipeline,
            'ECat': data_utils.ECat, 'EInt': data_utils.EInt, 'EFloat': data_utils.EFloat,
            'SequenceExtractor': data_utils.SequenceExtractor, 'SensorAugmentor': data_utils.SensorAugmentor,
            'pretraining_keras': pretraining, 'multibranch_keras': mb,
            'SKOPT_AVAILABLE': mb.HierarchicalBayesSearchCV is not None,
        }
        with redirect_stdout(StringIO()):
            for index in (3, 4, 11, 14):
                exec(compile(self.source(index), f'notebook-{index}', 'exec'), namespace)
        return namespace

    def test_source_candidates_share_finetuning_set_and_keep_reserved_sequences_out(self):
        X, y = recordings()
        fine_sets = []
        for source in ('non_target', 'all', 'target', 'baseline'):
            with self.subTest(source=source):
                estimator = pretraining.AdvancedMultiBranchSequenceClassifier(
                    extractor=data_utils.SequenceExtractor(pretraining_source=source),
                    validation_split=0.15, pretraining_validation_split=0.15,
                    random_state=42, primary_target='bfrb',
                )
                estimator.extractor_ = clone(estimator.extractor)
                fine, reserved = estimator._split_validation_sequences(X, y)
                fine_sets.append(set(fine.sequence_id))
                self.assertTrue(fine.is_target.eq(1).all())
                reserve_ids = set(estimator.validation_sequence_ids_)
                self.assertTrue(reserve_ids.isdisjoint(estimator.preprocessing_sequence_ids_))
                self.assertTrue(set(fine.sequence_id).isdisjoint(reserve_ids))
                if source == 'baseline':
                    self.assertTrue(estimator.pretraining_baseline_)
                    self.assertFalse(estimator.pretraining_completed_)
                    self.assertFalse(hasattr(estimator, '_pretraining_X'))
                    estimator.model_ = SimpleNamespace(optimizer=SimpleNamespace(
                        learning_rate=SimpleNamespace(assign=lambda value: None)))
                    with patch.object(estimator, '_supervised_pretrain', side_effect=AssertionError('unexpected pretraining')):
                        estimator._before_supervised_fit(time_series_shapes={}, static_shapes={}, sequence_length=3,
                                                         model_inputs=[], train_targets=[], sample_weight=None,
                                                         validation_data=None)
                else:
                    pretrain_ids = set(estimator._pretraining_X.sequence_id)
                    self.assertTrue(pretrain_ids.isdisjoint(reserve_ids))
                    expected_domain = ({0} if source == 'non_target' else {1} if source == 'target' else {0, 1})
                    self.assertEqual(set(estimator._pretraining_X.is_target), expected_domain)
        self.assertTrue(all(ids == fine_sets[0] for ids in fine_sets))

    @unittest.skipIf(mb.HierarchicalBayesSearchCV is None, 'scikit-optimize unavailable')
    def test_first_five_bayes_candidates_clone_and_decode_full_configuration(self):
        X, y = recordings()
        fitted = []

        def fake_fit(estimator, frame, labels, **kwargs):
            estimator.extractor_ = clone(estimator.extractor)
            estimator._validate_extractor()
            fine, _ = estimator._split_validation_sequences(frame, labels)
            assert not fine.empty
            assert estimator.bert_d_model % estimator.bert_num_heads == 0
            assert estimator.learning_rate == 1e-4
            assert estimator.pretraining_learning_rate == 1e-4
            assert estimator.use_lr_scheduler is False
            assert estimator.pretraining_use_lr_scheduler is False
            fitted.append((estimator.extractor.pretraining_source, estimator.bert_d_model))
            return estimator

        def fake_score(estimator, frame, labels, sample_weight=None):
            selected = estimator.extractor_.select_finetuning_sequences(frame, labels)
            assert not selected.empty and selected.is_target.eq(1).all()
            return 0.5 + float(estimator.bert_d_model) / 1000

        with TemporaryDirectory() as directory:
            namespace = self.namespace(directory)
            search = mb.HierarchicalBayesSearchCV(
                estimator=namespace['pipeline'], search_spaces=namespace['BAYESIAN_PARAM_SPACE'],
                n_iter=5, cv=GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42),
                random_state=42, n_jobs=1, scoring=mb.candidate_score, refit=False,
                return_train_score=True, error_score='raise',
            )
            with patch.object(pretraining.AdvancedMultiBranchSequenceClassifier, '_fit_impl', fake_fit), \
                    patch.object(pretraining.AdvancedMultiBranchSequenceClassifier, 'score', fake_score), \
                    redirect_stdout(StringIO()):
                search.fit(X, y, groups=X.subject)
        self.assertEqual(len(search.cv_results_['params']), 5)
        self.assertEqual(len(fitted), 5)
        self.assertTrue(np.all(np.asarray(search.cv_results_['mean_test_score']) >= 0.5))
        self.assertTrue({source for source, _ in fitted} <= {'non_target', 'all', 'target', 'baseline'})

    def test_holdout_labels_follow_surviving_prediction_ids_and_keep_row_alignment(self):
        # A quality filter can remove one entire sequence during inference. The
        # final notebook cell must evaluate the surviving predictions only.
        frame = pd.DataFrame({
            'sequence_id': ['keep_b', 'remove', 'keep_a', 'keep_b', 'remove', 'keep_a'],
            'is_target': [1] * 6, 'bfrb': ['b', 'bad', 'a', 'b', 'bad', 'a'],
        }, index=[31, 12, 18, 32, 13, 19])
        predictions = pd.Series(['a', 'b'], index=pd.Index(['keep_a', 'keep_b'], name='sequence_id'))
        extractor = SimpleNamespace(select_finetuning_sequences=lambda X, y: X.copy())
        estimator = SimpleNamespace(extractor_=extractor)
        model = SimpleNamespace(named_steps={'estimator': estimator}, predict=lambda X: predictions)
        namespace = {'experiment_succeeded': True, 'search': SimpleNamespace(best_estimator_=model),
                     'X_test': frame, 'y_test': frame.copy(), 'TARGET_COL': 'bfrb',
                     'evaluate_holdout': data_utils.evaluate_holdout, 'json': json, 'pd': pd}
        with TemporaryDirectory() as directory:
            namespace.update(results_dir=Path(directory), timestamp='test', experiment_name='pretraining')
            with redirect_stdout(StringIO()):
                exec(compile(self.source(22), 'holdout-cell', 'exec'), namespace)
        results = namespace['eval_results']['results_df']
        self.assertEqual(set(results.sequence_id), {'keep_a', 'keep_b'})
        self.assertFalse(results.bfrb_pred.isna().any())
        self.assertEqual(namespace['eval_results']['gesture_f1'], 1.0)


if __name__ == '__main__':
    unittest.main()

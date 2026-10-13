"""Failure reporting and safe refitting for unattended multibranch searches."""
import time
import warnings
from collections import Counter

import numpy as np
from sklearn.base import clone
from base_utils_sol import GASearchCV
from multibranch_candidate_rules import candidate_error, canonical_candidate

try:
    from skopt import BayesSearchCV
except ImportError:
    BayesSearchCV = None


class _CandidateRules:
    """Shared dependency checking and canonical cache keys, independent of TensorFlow."""
    _proposal_attempts = 32

    def _initialise_rules(self, y, fit_params):
        self._base_params_ = self.estimator.get_params(deep=True)
        self._rules_y_ = y
        self._phase_labels_available_ = any(
            (key == 'phase_labels' or key.endswith('__phase_labels')) and value is not None
            for key, value in fit_params.items()
        )
        self._label_validation_cache_ = {}
        self.rejected_candidate_count_ = 0
        self.rejected_candidate_reasons_ = Counter()
        self.reused_candidate_count_ = 0

    def _active_params(self, params):
        return canonical_candidate(params, getattr(self, '_base_params_', self.estimator.get_params(deep=True)))

    def _candidate_error(self, params):
        return candidate_error(params, getattr(self, '_base_params_', self.estimator.get_params(deep=True)),
                               y=getattr(self, '_rules_y_', None),
                               phase_labels_available=getattr(self, '_phase_labels_available_', False),
                               label_cache=getattr(self, '_label_validation_cache_', None))

    def _reject(self, reason):
        self.rejected_candidate_count_ += 1
        self.rejected_candidate_reasons_[reason] += 1

    def _penalty(self):
        return float(self._base_params_.get('estimator__invalid_candidate_score', -1.0))


class HierarchicalGASearchCV(_CandidateRules, GASearchCV):
    """Evolve full genomes; train/cache only their active architecture settings."""
    def fit(self, X, y=None, groups=None, **fit_params):
        self._initialise_rules(y, fit_params)
        return super().fit(X, y, groups=groups, **fit_params)

    def _on_candidate_cache_hit(self):
        self.reused_candidate_count_ += 1

    def _sample_individual(self, space, rng):
        for _ in range(self._proposal_attempts):
            proposal = super()._sample_individual(space, rng)
            reason = self._candidate_error(proposal)
            if reason is None:
                return proposal
            self._reject(reason)
        # An impossible user space must terminate, with a recorded penalty and no model fit.
        return proposal

    def _mutate(self, individual, space, rng):
        for _ in range(self._proposal_attempts):
            proposal = super()._mutate(individual, space, rng)
            reason = self._candidate_error(proposal)
            if reason is None:
                return proposal
            self._reject(reason)
        return self._sample_individual(space, rng)

    def _canonical_key(self, individual):
        return super()._canonical_key(self._active_params(individual))

    def _evaluate_individual(self, params, X, y, splits, fit_params):
        active = self._active_params(params)
        reason = self._candidate_error(params)
        if reason is not None:
            self._reject(reason)
            penalty = self._penalty()
            count = len(splits)
            return dict(params=active, fit_times=[0.0]*count, score_times=[0.0]*count,
                        test_scores=[penalty]*count, train_scores=[penalty]*count,
                        mean_test_score=penalty, std_test_score=0.0,
                        mean_train_score=penalty, std_train_score=0.0, candidate_error=reason)
        result = super()._evaluate_individual(active, X, y, splits, fit_params)
        result['candidate_error'] = None
        return result

    def _build_cv_results(self, records, maximize):
        results = super()._build_cv_results(records, maximize)
        # Different backbones have different active genes; include their union in the CSV.
        for key in sorted({key for record in records for key in record['params']}):
            results[f'param_{key}'] = [record['params'].get(key) for record in records]
        results['candidate_error'] = [record.get('candidate_error') for record in records]
        return results


if BayesSearchCV is not None:
    class HierarchicalBayesSearchCV(_CandidateRules, BayesSearchCV):
        """Reject incompatible proposals before CV and reuse equivalent architecture fits."""
        def fit(self, X, y=None, groups=None, callback=None, **fit_params):
            self._initialise_rules(y, fit_params)
            self._effective_score_cache_ = {}
            self._exhausted_search_spaces_ = set()
            return super().fit(X, y, groups=groups, callback=callback, **fit_params)

        def _step(self, search_space, optimizer, *args, n_points=1):
            # skopt 0.10 supplies score_name; earlier versions only supplied the evaluator.
            modern = len(args) == 2
            score_name, evaluate_candidates = args if modern else ('mean_test_score', args[0])
            last_result = None
            for _ in range(n_points):
                rejected_points = []
                attempts = 1 if id(search_space) in self._exhausted_search_spaces_ else self._proposal_attempts
                for attempt in range(attempts):
                    proposed_point = (optimizer.ask(n_points=1)[0] if attempt == 0 else
                                      optimizer.space.rvs(n_samples=1, random_state=optimizer.rng)[0])
                    point = [np.asarray(value).item() for value in proposed_point]
                    proposal = dict(zip(sorted(search_space), point))
                    reason = self._candidate_error(proposal)
                    if reason is None:
                        break
                    self._reject(reason)
                    if attempt < attempts - 1:
                        rejected_points.append(point)
                if reason is not None:
                    self._exhausted_search_spaces_.add(id(search_space))
                active = self._active_params(proposal)
                key = GASearchCV._canonical_key(self, active)
                if reason is None and key in self._effective_score_cache_:
                    score = self._effective_score_cache_[key]
                    self.reused_candidate_count_ += 1
                else:
                    # Even an entirely invalid space gets a CV row through the estimator's
                    # finite-penalty path, allowing the existing all-failed reporting flow.
                    results = evaluate_candidates([active])
                    if score_name is None:
                        names = [name for name in results if name.startswith('mean_test_')]
                        if len(names) != 1:
                            raise ValueError('Hierarchical Bayesian search requires one scoring metric.')
                        score_name = names[0]
                    score = float(results[score_name][-1])
                    if not np.isfinite(score):
                        score = self._penalty()
                    if reason is None:
                        self._effective_score_cache_[key] = score
                # One optimizer update per step avoids repeatedly fitting its surrogate
                # while rejecting several incompatible proposals.
                last_result = optimizer.tell(rejected_points + [point],
                                             [-self._penalty()]*len(rejected_points) + [-score])
            return (last_result, score_name) if modern else last_result
else:
    HierarchicalBayesSearchCV = None


def candidate_score(estimator, X, y):
    """Finite penalties keep Bayesian optimisation running even if all fits fail."""
    model = estimator.named_steps['estimator']
    if getattr(model, 'fit_failed_', False):
        return float(model.invalid_candidate_score)
    try:
        return float(estimator.score(X, y))
    except Exception as exc:
        if model.invalid_candidate_policy == 'raise':
            raise
        model.fit_error_ = f'{type(exc).__name__}: {exc}'
        model._record_candidate_failure('score', model.fit_error_)
        warnings.warn(f'Candidate scoring skipped: {model.fit_error_}', RuntimeWarning)
        return float(model.invalid_candidate_score)


def refit_valid_candidate(search, X, y, **fit_params):
    """Try successful CV candidates in score order; never refit a penalty winner."""
    results = search.cv_results_
    score_key = 'mean_test_score' if 'mean_test_score' in results else 'mean_test_quality'
    scores = np.asarray(results[score_key], dtype=float)
    split_keys = [k for k in results if k.startswith('split') and
                  (k.endswith('_test_score') or k.endswith('_test_quality'))]
    valid = np.isfinite(scores) & (scores >= 0)
    for key in split_keys:
        valid &= np.isfinite(results[key]) & (np.asarray(results[key]) >= 0)
    search.refit_errors_ = []
    search.best_estimator_ = None
    search.best_index_ = None
    for idx in np.argsort(-scores, kind='stable'):
        if not valid[idx]:
            continue
        estimator = clone(search.estimator).set_params(**results['params'][idx])
        estimator.set_params(estimator__invalid_candidate_policy='raise')
        start = time.perf_counter()
        try:
            estimator.fit(X, y, **fit_params)
        except Exception as exc:
            search.refit_errors_.append({'candidate': int(idx), 'error': f'{type(exc).__name__}: {exc}'})
            continue
        search.best_estimator_ = estimator
        search.best_index_ = int(idx)
        search.best_params_ = dict(results['params'][idx])
        search.best_score_ = float(scores[idx])
        search.refit_time_ = time.perf_counter() - start
        return True
    search.best_score_ = np.nan
    search.best_params_ = {}
    return False

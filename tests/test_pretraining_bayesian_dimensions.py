"""Exercise range dimensions through optimization, CV fitting, and refitting."""
import json
import unittest

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

import test_pretraining_large_notebook as notebook


class PolynomialRidge(RegressorMixin, BaseEstimator):
    """Small real model accepting a JSON category like the notebook estimators."""

    fit_count = 0

    def __init__(self, degree=2, alpha=0.1, configuration=None):
        self.degree = degree
        self.alpha = alpha
        self.configuration = configuration

    def _features(self, X):
        return np.column_stack([
            X[:, column] ** power
            for column in range(X.shape[1])
            for power in range(1, int(self.degree) + 1)
        ])

    def fit(self, X, y):
        self.configuration_ = json.loads(self.configuration)
        self.model_ = Ridge(
            alpha=self.alpha,
            fit_intercept=self.configuration_["fit_intercept"],
        ).fit(self._features(X), y)
        type(self).fit_count += 1
        return self

    def predict(self, X):
        return self.model_.predict(self._features(X))


@unittest.skipIf(notebook.mb.HierarchicalBayesSearchCV is None,
                 "scikit-optimize unavailable")
class BayesianRangeSearchTests(unittest.TestCase):
    def test_integer_real_and_structured_categories_fit_after_initial_exploration(self):
        from skopt.learning import GaussianProcessRegressor
        from skopt.space import Categorical, Integer, Real

        configurations = [{"fit_intercept": True}, {"fit_intercept": False}]
        space = notebook.mb.prepare_multibranch_bayesian_space({
            "degree": Integer(1, 3),
            "alpha": Real(1e-4, 1.0, prior="log-uniform"),
            "configuration": configurations,
        })
        self.assertIsInstance(space["degree"], Integer)
        self.assertIsInstance(space["alpha"], Real)
        self.assertEqual(space["alpha"].prior, "log-uniform")
        self.assertIsInstance(space["configuration"], Categorical)
        self.assertEqual([json.loads(value) for value in
                          space["configuration"].categories], configurations)

        rng = np.random.default_rng(23)
        X = rng.uniform(-1.0, 1.0, size=(30, 2))
        y = 1.5 * X[:, 0] + 0.6 * X[:, 1] ** 2 + 0.2
        PolynomialRidge.fit_count = 0
        search = notebook.mb.HierarchicalBayesSearchCV(
            PolynomialRidge(), search_spaces=space, n_iter=6, cv=3,
            n_jobs=1, random_state=42, verbose=0, refit=True,
            scoring="neg_mean_squared_error", error_score="raise",
            optimizer_kwargs={
                "n_initial_points": 2,
                "model_queue_size": 1,
                "acq_optimizer": "sampling",
                "acq_optimizer_kwargs": {"n_points": 32},
            },
        ).fit(X, y)

        parameters = search.cv_results_["params"]
        self.assertEqual(len(parameters), 6)
        self.assertEqual(PolynomialRidge.fit_count, 6 * 3 + 1)
        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"]).all())
        for candidate in parameters:
            self.assertIsInstance(candidate["degree"], (int, np.integer))
            self.assertNotIsInstance(candidate["degree"], (bool, np.bool_))
            self.assertGreaterEqual(candidate["degree"], 1)
            self.assertLessEqual(candidate["degree"], 3)
            self.assertIsInstance(candidate["alpha"], (float, np.floating))
            self.assertGreaterEqual(candidate["alpha"], 1e-4)
            self.assertLessEqual(candidate["alpha"], 1.0)
            self.assertIn(json.loads(candidate["configuration"]), configurations)

        # More candidates than initial points require the fitted GP ask/tell path.
        self.assertEqual(len(search.optimizers_), 1)
        optimizer = search.optimizers_[0]
        self.assertEqual(len(optimizer.yi), 6)
        self.assertLess(optimizer._n_initial_points, 0)
        self.assertEqual(len(optimizer.models), 1)
        self.assertIsInstance(optimizer.models[0], GaussianProcessRegressor)
        self.assertEqual(optimizer.models[0].X_train_.shape[0], 6)
        self.assertEqual(search.best_estimator_.configuration_,
                         json.loads(search.best_params_["configuration"]))
        self.assertTrue(np.isfinite(search.predict(X)).all())
        self.assertLess(np.mean((search.predict(X) - y) ** 2), 0.02)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from typing import Any

import utils_hierarchy_position_aug as base_utils


class DecodingKerasMultiBranchClassifier(base_utils.KerasMultiBranchClassifier):
    """KerasMultiBranchClassifier that accepts JSON-encoded branch dicts from BayesSearchCV."""

    _BRANCH_KEYS = {"branch_config", "branch_filters", "branch_kernel_sizes", "branch_pool_sizes"}

    def set_params(self, **params: Any):
        decoded: dict[str, Any] = {}
        for key, value in params.items():
            if key in self._BRANCH_KEYS and isinstance(value, str):
                try:
                    decoded[key] = json.loads(value)
                except Exception:
                    decoded[key] = value
            else:
                decoded[key] = value
        return super().set_params(**decoded)


def prepare_branch_param_space(param_space: dict[str, Any], search_mode: str, Categorical=None) -> dict[str, Any]:
    """Encode branch dict lists for skopt BayesSearchCV while leaving grid lists alone."""
    if search_mode != "bayesian":
        return param_space
    if Categorical is None:
        raise ValueError("Categorical must be supplied for bayesian search spaces.")

    branch_suffixes = (
        "__branch_config",
        "__branch_filters",
        "__branch_kernel_sizes",
        "__branch_pool_sizes",
    )

    out = dict(param_space)
    for key, value in list(out.items()):
        if key.endswith(branch_suffixes) and isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                out[key] = Categorical([json.dumps(v, sort_keys=True) for v in value])
    return out

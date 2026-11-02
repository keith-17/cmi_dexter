# NOOP validation: original pretraining notebook

The change lives on multibranch/v2/develop. The active notebook is
notebooks/multibranch_pretraining.ipynb. V2 refers only to the branch.

## Numerical and model verification

- 48 focused tests were discovered in the lightweight runtime: 43 passed and
  5 neural checks were skipped there. Those paths were tested in the working
  TensorFlow runtime. See kaggle_build/noop_preflight/native_noop_tests.log.
- 68 focused checks passed in the working TensorFlow runtime, including real
  fitting, validation, prediction, all four temporal backbones, supervised
  pretraining, masked SSL and actual-network permutation importance.
  See tmp/noop_integration_regressions.log.
- Final search-helper checks passed 18 tests, including object replacement,
  inactive-setting reuse and a 10^24-combination model-grid scalability case.
  See tmp/noop_search_final.log. These counts overlap.
- A 1,200-row real-data probe verified exact parent/child tensors with NOOP off
  and finite aligned summaries/support values with the generous options on.
  See kaggle_build/noop_preflight/real_data_extraction.json.

Disabled-mode tests also compare architecture, learned weights and predictions
under the same seed. Missing-data, padding, timing, PSD scaling, quaternion
units, harmonic ambiguity, cropping and augmentation alignment are checked.

## Broad-suite verification and corrected expectations

The full working-TensorFlow run exercised 411 tests. It found seven outdated
expectations about notebook configuration, independent search spaces and
callback validation data. Those test expectations were corrected; all 27
selected regression tests then passed, including every initially failing case.
The neural implementation did not require further changes. This is a full
suite followed by targeted rechecks, not a claim of one clean full-suite run.

Logs: kaggle_build/noop_preflight/native_full_tensorflow_tests.log and
kaggle_build/noop_preflight/native_corrected_regressions.log. The coverage audit
is kaggle_build/noop_preflight/native_final_test_audit.json.

The Conda interpreter needs its own Library/bin on process PATH for LAPACK.
Without it, an independent scipy.linalg.lstsq call aborted; setting the proper
process PATH resolved that environment issue.

## Native notebook experiment checks

BEST_PARAMS contains the 215 discussed winning settings and NOOP-off defaults.
It is no longer overwritten by first Grid choices. The one-candidate preset
preserves those baseline settings and adds only NOOP controls. Ordinary Grid,
Bayesian and evolutionary ranges retain their editable declarations.

The existing scripts/kaggle_experiment.py packager is reused. Baseline retains
all 31 notebook cells exactly plus the source-loading bootstrap. Enhanced
changes only the search_mode assignment to grid. Both execute native
GridSearchCV, CV, refit and holdout evaluation; no alternate classifier factory
or fit interception is used. Source, version, private T4 shape and pinned image
are verified through the API. Live plots distinguish CV from refit.

The initial wrapper runs were superseded, and Kaggle acknowledged cancellation.
Their receipts are an audit trail, not results of the corrected comparison.
Corrected receipts are in experiments/noop_candidate21/submitted_notebook_20261102.json.

Permutation importance is the final notebook section and uses only saved
internal-validation sequences in the fitted final-task domain. Final scores
remain sequence-disjoint rather than participant-disjoint. The configured
100 Hz and chosen frequency bands remain explicit analysis assumptions.

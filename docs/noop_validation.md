# NOOP V2 validation record (2 November 2026)

The change is on `multibranch/v2/develop`, based on
`4a87762f80634a251dbcbf8586d8ed4fd4449101`. Source files in the synced ChatGPT
project were not modified.

## Checks completed before the matched GPU submissions

- New NOOP tests: 48 discovered, 43 passed, 5 neural tests skipped in the
  lightweight Python environment. Log: `kaggle_build/noop_preflight/noop_unittest.log`.
- A separate working TensorFlow environment passed 68 focused checks, including
  actual supervised fitting, validation, prediction, all four temporal model
  paths, supervised pretraining, masked SSL and trained-network importance.
  Log: `tmp/noop_integration_regressions.log`.
- Final search checks passed 18 tests, including nested option replacement,
  conditional Bayesian/evolutionary reuse and a grid whose untouched model
  axes contain 10^24 combinations. Log: `tmp/noop_search_final.log`.
- A 1,200-row sample from the real CMI CSV produced exact parent/child output
  equality with NOOP off, and finite aligned feature/support values with all
  selected methods enabled. Record:
  `kaggle_build/noop_preflight/real_data_extraction.json`.
- The full real-data preparation reproduced the historical checksum and every
  saved train/holdout sequence ID: 8,151 input sequences, 50 excluded for wholly
  invalid quaternions, 4,050 outer train and 4,051 holdout sequences; 2,539
  Target holdout sequences. Record:
  `kaggle_build/noop_preflight/data_verification.json`.
- The new notebook's code cells compile and its schema/search wiring is checked.
  Existing non-NOOP search ranges and original notebooks are preserved.

The checks above overlap; their counts should not be added together.

## Existing broad-suite failures

The starting commit was extracted into an isolated directory and tested with
the same lightweight interpreter. Its broad suite already reports 2 failures
and 29 errors: stale expectations about older notebook cells and unavailable
local TensorFlow functionality. Logs are
`kaggle_build/noop_preflight/original_full_unittest.log` and
`kaggle_build/noop_preflight/full_unittest_final.log`. The final modified suite
ran 405 tests; the original ran 357. Their failing test names are identical:
there are no additional failures or errors. These results are not described
as a passing full suite. Actual neural checks use the separate working
TensorFlow environment.

Invoking the Conda interpreter directly also required its `Library/bin` on
process PATH: without it, an independent `scipy.linalg.lstsq` call aborted
before Python could raise an exception. Setting the correct process PATH
resolved that environment issue; no numerical algorithm was changed for it.

## Corrected notebook runs

The active notebook is `notebooks/multibranch_pretraining.ipynb`. V2 is only the
branch name. The initial wrapper runs were superseded and cancellation was
acknowledged; their receipts remain an audit trail, not comparison results.

The original notebook now contains the complete discussed BEST_PARAMS baseline,
optional NOOP controls in all three searches, a one-candidate NOOP Grid preset,
and the final internal-validation importance section. The corrected native
NOOP checks discovered 48 tests: 43 passed and 5 actual neural tests skipped in
the lightweight runtime. The working TensorFlow checks above verify those
neural paths separately. Updated legacy notebook tests select cells by meaning
rather than obsolete cell numbers.

The experiment packager reuses scripts/kaggle_experiment.py and preserves actual
notebook cells, adding only its existing source bootstrap. Baseline uses best
params; enhanced changes only search_mode to grid, enabling the visible single
candidate preset. Both retain native CV/refit/holdout evaluation. Live epoch
observations distinguish CV from refit and never mix their histories.

Source hashes, private T4 configuration and notebook versions are verified
through the Kaggle API. Collection checks the native one-candidate CSV/summary,
final 2,000-epoch refit and saved permutation importance. No monitor resubmits
training.

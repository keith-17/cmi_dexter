# Candidate21 baseline and one NOOP Grid candidate

V2 refers only to the branch `multibranch/v2/develop`. The active notebook is
`notebooks/multibranch_pretraining.ipynb`; no separate V2 notebook is used.

Its visible `BEST_PARAMS` now contains all 215 discussed winning settings,
including 2,000 epochs, no early stopping, final weights, no augmentation,
raw acceleration and 250-sample chunks. NOOP families are None in this baseline.
Later search cells no longer overwrite this dictionary with first grid choices.

Run the notebook with `search_mode = 'best params'` for baseline. Then use
`search_mode = 'grid'` and `single_noop_experiment = True` for exactly one
candidate: the same baseline plus the visible `NOOP_EXPERIMENT_PARAMS` preset.
Set the latter flag False to restore the declared ordinary Grid ranges.
Bayesian and evolutionary spaces retain their editable NOOP controls.

Both runs execute the notebook's existing GridSearchCV -> validation runner ->
CV/refit/holdout flow. One parameter candidate still means a CV fit and the
notebook's usual final refit. No replacement classifier factory, fit interception
or external training runtime is used. Permutation importance is the final
notebook section and uses the fitted internal-validation domain.

`scripts/noop_comparison.py` reuses the existing GPU source packager. It includes
every actual notebook cell plus the source bootstrap. The baseline package
retains those cells exactly; the enhanced package changes only the search_mode
assignment. Source/version receipts are preserved and results use the native
experiments CSV and summary JSON. Live curves observe existing Keras log output;
they distinguish the CV and refit stages.

The earlier wrapper submissions were cancelled following the user's correction.
`submitted_20261102.json` records that superseded source, and
`superseded_wrapper_runs.json` records cancellation acknowledgements. Their
partial values are not results of the corrected notebook comparison.

The historical expected_split.json is reference material, not a replacement
for the native notebook's data preparation. Both corrected runs use that same
native preparation and seed. Results remain sequence-disjoint rather than
participant-disjoint. The configured 100 Hz is retained, not independently
measured. Selected frequency bands are exploratory settings.

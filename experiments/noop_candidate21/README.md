# Frozen Candidate21 baseline and generous NOOP comparison

The two private GPU runs each fit once for 2,000 epochs. They preserve all 215
effective settings from the completed winning Candidate21 configuration, with
no augmentation, no early stopping and no best-weight restoration. Both use the
new child extractor: its all-None options follow the original extraction path;
the enhanced run adds the methods in `enhanced_options.json` and their static
input branch. This is a combined-additions comparison, not a per-feature ablation.

`reference_params.json` comes from the verified completed Candidate21 parameter
record. `expected_split.json` freezes the exact original data checksum, 50 wholly
invalid-quaternion exclusions and balanced seed42 membership. Both runs retain
all orientations, split before Target-domain selection, and evaluate the same
2,539 Target sequences once with final weights. These are sequence-disjoint,
not participant-disjoint, results; Target-only binary F1 does not establish
Target/non-Target detection. The original configured 100 Hz remains unchanged;
this comparison does not independently establish the hardware sampling rate.

The generous option set adds all implemented motion summaries, signed corrected
acceleration autocorrelation with harmonic checks, all spectral descriptors and
quality features. Its 0.5-15 Hz band, two-cycle minimum and two-sample interpolation
limit are explicit exploratory settings, not known BFRB physiological constants.
Unsupported slow periodicities on short windows remain flagged. Quality modes
do not remove windows or alter the frozen scoring population.

The two runs use the same source snapshot, Kaggle image, T4 shape and seed42
policy. An extra NOOP branch necessarily changes the enhanced model's parameter
count; GPU/initialization variation can still affect the measured difference.
Each run saves full epoch telemetry, curves, final/best weights, one holdout
evaluation, internal-validation permutation importance, and provenance hashes.
Importance never uses the final holdout and never changes the fitted weights.

`scripts/noop_comparison.py` prepares and submits each variant once, preserves
attempt/acknowledgement receipts, verifies the remote source, monitors API logs,
and collects non-weight outputs. `scripts/plot_noop_comparison.py` plots only
observed epochs from live telemetry or completed histories. No monitor submits
or retries training.

"""Run one frozen Candidate21 baseline or NOOP fit on Kaggle.

This module is packaged with the source snapshot. It never submits another run.
The final holdout is evaluated once; permutation importance uses internal validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiment_results"
STATE = {"status": "STARTING", "fit_count": 0, "holdout_evaluation_count": 0}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def update(**values):
    STATE.update(values)
    STATE["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUTPUT / "execution_state.json", STATE)
    print("NOOP_STATE " + json.dumps(STATE, default=str), flush=True)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_data(data_path, data_utils, *, compact=False):
    """Reproduce the completed Candidate21 pool and exact balanced seed42 split."""
    import numpy as np
    import pandas as pd
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    sensors = [c for c in header if c.startswith(("acc_", "rot_", "tof_", "thm_"))]
    columns = None
    if compact:
        columns = ["sequence_id", "subject", "gesture", "sequence_type", "orientation",
                   "acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    frame = pd.read_csv(data_path, usecols=columns,
                        dtype={c: np.float32 for c in sensors if columns is None or c in columns})
    input_rows, input_sequences = len(frame), frame["sequence_id"].nunique()
    frame["gesture"] = frame["gesture"].fillna("non_bfrb").astype(str)
    frame["orientation"] = frame["orientation"].fillna("Unknown").astype(str)
    frame["is_target"] = frame["sequence_type"].eq("Target").astype(np.uint8)
    frame["bfrb"] = frame["gesture"].where(frame["is_target"].eq(1), "non_bfrb")
    parts = {g: g.split(" - ") for g in frame["gesture"].unique()}
    frame["gesture_position"] = frame["gesture"].map({g: p[0] for g, p in parts.items()})
    frame["gesture_action"] = frame["gesture"].map({g: p[-1] for g, p in parts.items()})
    q = frame[["rot_w", "rot_x", "rot_y", "rot_z"]].to_numpy(dtype=np.float64)
    valid = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0)
    available = pd.Series(valid, index=frame.index).groupby(frame["sequence_id"]).any()
    excluded = sorted(map(str, available.index[~available]))
    frame = frame.loc[~frame["sequence_id"].isin(excluded)].copy()
    train, test = data_utils.sample_balanced_split(frame, train_pct=.5, test_pct=.5, random_state=42)
    train_ids = sorted(map(str, train["sequence_id"].unique()))
    test_ids = sorted(map(str, test["sequence_id"].unique()))
    assert set(train_ids).isdisjoint(test_ids)
    labels = [c for c in frame if c not in sensors and c not in {"row_id", "sequence_counter"}]
    y_train, y_test = train[labels].copy(), test[labels].copy()
    manifest = dict(orientation="ALL", target_only_before_split=False,
        problematic_filter_enabled=False, train_csv_sha256=file_hash(data_path),
        input_rows=int(input_rows), input_sequences=int(input_sequences),
        filtered_sequences=int(frame["sequence_id"].nunique()),
        train_sequences=len(train_ids), holdout_sequences=len(test_ids),
        target_train_sequences=int(train.drop_duplicates("sequence_id")["is_target"].sum()),
        target_holdout_sequences=int(test.drop_duplicates("sequence_id")["is_target"].sum()),
        train_ids=train_ids, holdout_ids=test_ids,
        excluded_sequences=len(excluded), excluded_ids=excluded)
    return train, y_train, test, y_test, manifest


def make_pipeline(config, reference):
    from sklearn.pipeline import Pipeline
    import base_utils as bu
    import pretraining_utils as pu
    from noop_sequence_extractor import NoopOptions, NoopSequenceExtractor
    noop = NoopOptions(**config["noop_options"])
    extractor = NoopSequenceExtractor(noop=noop, output_format="chunks",
        stft_configs=[{}], cwt_configs=[{}], filter_problematic_sequences=False,
        ideal_skew_threshold=1.6, problematic_features_threshold=7,
        problematic_feature_cols=["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_w", "rot_z"])
    estimator = pu.AdvancedMultiBranchSequenceClassifier(
        extractor=extractor,
        augmentor=bu.SensorAugmentor(sequence_col="sequence_id", counter_col="sequence_counter",
            prob=0., per_aug_prob=0., diagnostics=False),
        primary_target="bfrb", pretraining_validation_group_column=None,
        validation_group_column=None, temporal_architecture="bert", use_bigru=False,
        use_orientation_aux_head=False, use_phase_aux_head=False, custom_aux_heads=[],
        orientation_loss_weight=.2, phase_loss_weight=.2, phase_label_provider=None,
        invalid_candidate_policy="penalize", candidate_failure_log=None, verbose=2, random_state=42)
    pipeline = Pipeline([("estimator", estimator)])
    pipeline.set_params(**reference)
    effective = pipeline.get_params(deep=True)
    for name, value in reference.items():
        assert effective[name] == value, name
    assert estimator.epochs == 2000 and estimator.early_stopping_patience == 0
    assert estimator.restore_best_weights is False
    assert estimator.augmentor.prob == 0 and estimator.extractor.pretraining_source == "baseline"
    assert estimator.validation_group_column is None and estimator.pretraining_validation_group_column is None
    return pipeline


def main():
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    sys.path.insert(0, str(ROOT / "src"))
    import base_utils as bu
    from pretraining_search import _holdout_results
    from plot_noop_comparison import save_run_curves
    from noop_importance import compute_permutation_importance

    config = json.loads((ROOT / "assets/config.json").read_text())
    reference = json.loads((ROOT / "assets/reference_params.json").read_text())
    expected = json.loads((ROOT / "assets/expected_split.json").read_text())
    provenance = json.loads((ROOT / "assets/provenance.json").read_text())
    OUTPUT.mkdir(exist_ok=True)
    for name, digest in provenance["source_sha256"].items():
        assert file_hash(ROOT / "src" / name) == digest, name
    for name, digest in provenance["runtime_sha256"].items():
        assert file_hash(ROOT / name) == digest, name
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "provenance.json", provenance)
    write_json(OUTPUT / "reference_params.json", reference)
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "GPU was not allocated; no fit attempted."
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.utils.set_random_seed(42)
    with tf.device("/GPU:0"):
        probe = tf.linalg.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
    assert "GPU" in probe.device
    write_json(OUTPUT / "runtime.json", dict(python=sys.version, tensorflow=tf.__version__,
        numpy=np.__version__, pandas=pd.__version__, gpu=tf.config.experimental.get_device_details(gpus[0])))
    update(status="PREPARING_DATA", variant=config["variant"], gpu_verified=True, requested_epochs=2000)
    X_train, y_train, X_test, y_test, manifest = prepare_data(bu.find_data_root() / "train.csv", bu)
    assert manifest == expected, "Data bytes or frozen original sequence membership changed."
    write_json(OUTPUT / "split_manifest.json", manifest)
    pipeline = make_pipeline(config, reference)
    estimator = pipeline.named_steps["estimator"]
    write_json(OUTPUT / "effective_params.json", {
        k: v for k, v in pipeline.get_params(deep=True).items()
        if k in reference or "__noop__" in k})

    def weights_hash(model):
        h = hashlib.sha256()
        for variable in model.weights:
            array = variable.numpy()
            h.update(str(array.shape).encode()); h.update(str(array.dtype).encode())
            h.update(array.tobytes(order="C"))
        return h.hexdigest()

    observed_history = {}
    class Telemetry(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.monotonic()

        def on_epoch_end(self, epoch, logs=None):
            values = {k: float(v) for k, v in (logs or {}).items()}
            assert values and all(np.isfinite(v) for v in values.values())
            values.setdefault("learning_rate", float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
            for k, v in values.items(): observed_history.setdefault(k, []).append(v)
            event = dict(epoch=epoch + 1, variant=config["variant"], **values)
            event["duration_seconds"] = time.monotonic() - self.epoch_start
            with (OUTPUT / "epoch_telemetry.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print("NOOP_EPOCH " + json.dumps(event), flush=True)
            if (epoch + 1) % 100 == 0 or epoch == 0:
                write_json(OUTPUT / "history_partial.json", observed_history)
                save_run_curves(observed_history, OUTPUT / "accuracy_loss_partial.png",
                                title=f"{config['variant']} - partial")
            if epoch + 1 == 2000:
                update(status="FINAL_EPOCH_FINISHED", epochs_ran=2000,
                       final_epoch_weights_sha256=weights_hash(self.model))

    original_callbacks = estimator._build_callbacks
    def callbacks(has_validation_data=False):
        result = original_callbacks(has_validation_data=has_validation_data)
        assert len(result) == 1 and isinstance(result[0], tf.keras.callbacks.ReduceLROnPlateau)
        plateau = result[0]
        assert plateau.patience == 50 and plateau.factor == .5 and plateau.min_lr == 1e-7
        result += [tf.keras.callbacks.ModelCheckpoint(str(OUTPUT / "best_validation_loss.weights.h5"),
                   monitor="val_loss", save_best_only=True, save_weights_only=True),
                   Telemetry(), tf.keras.callbacks.CSVLogger(str(OUTPUT / "keras_history.csv"))]
        return result
    estimator._build_callbacks = callbacks
    original_fit = tf.keras.Model.fit
    def one_fit(model, *args, **kwargs):
        assert STATE["fit_count"] == 0 and kwargs["epochs"] == 2000
        update(status="TRAINING", fit_count=1)
        return original_fit(model, *args, **kwargs)
    tf.keras.Model.fit = one_fit
    try:
        pipeline.fit(X_train, y_train)
    finally:
        tf.keras.Model.fit = original_fit
    assert not getattr(estimator, "fit_failed_", False), getattr(estimator, "fit_error_", None)
    history = estimator.get_history_dict()
    assert len(history["loss"]) == 2000
    update(status="TRAINING_COMPLETE", training_complete=True, epochs_ran=2000)
    write_json(OUTPUT / "history.json", history)
    save_run_curves(history, OUTPUT / "accuracy_loss.png", title=config["variant"])
    estimator.model_.save_weights(OUTPUT / "final_epoch_2000.weights.h5")
    (OUTPUT / "model_architecture.json").write_text(estimator.model_.to_json(), encoding="utf-8")
    assert weights_hash(estimator.model_) == STATE["final_epoch_weights_sha256"]
    update(status="EVALUATING_HOLDOUT", holdout_evaluation_count=1)
    evaluation = _holdout_results(SimpleNamespace(best_estimator_=pipeline), X_test, y_test, "bfrb")
    exclusions = evaluation["exclusions"]
    assert exclusions["evaluated_sequences"] == exclusions["requested_sequences"] == 2539
    assert exclusions["sequence_ids"] == []
    predictions = evaluation.pop("results_df", None)
    if predictions is not None: predictions.to_csv(OUTPUT / "holdout_predictions.csv", index=False)
    write_json(OUTPUT / "holdout_metrics.json", evaluation)
    assert weights_hash(estimator.model_) == STATE["final_epoch_weights_sha256"]

    update(status="FEATURE_IMPORTANCE", holdout_complete=True, holdout_metrics=evaluation)
    validation_ids = set(estimator.validation_sequence_ids_)
    reserved = X_train.loc[X_train["sequence_id"].isin(validation_ids)]
    # Advanced validation reserves raw Target+Non-Target recordings before the
    # final-task domain adapter; importance must use the fitted target domain.
    internal = estimator.extractor_.select_finetuning_sequences(reserved, y_train)
    internal_y = y_train.loc[internal.index]
    assert set(internal["sequence_id"]).isdisjoint(set(X_test["sequence_id"]))
    importance = compute_permutation_importance(pipeline, internal, internal_y,
        repeats=3, random_state=42, include_noop_individual=True,
        output_dir=OUTPUT, prefix="validation_permutation", require_internal_validation=True)
    write_json(OUTPUT / "importance_summary.json", dict(importance.attrs))
    assert weights_hash(estimator.model_) == STATE["final_epoch_weights_sha256"]
    write_json(OUTPUT / "training_summary.json", dict(
        variant=config["variant"], epochs_ran=2000, fit_count=STATE["fit_count"],
        holdout_evaluation_count=STATE["holdout_evaluation_count"],
        final_metrics={k: float(v[-1]) for k, v in history.items()},
        best_validation_loss_epoch=int(np.argmin(history["val_loss"])) + 1,
        validation_sequence_count=len(validation_ids),
        noop_feature_names=getattr(estimator.extractor_, "noop_feature_names_", []),
        importance_rows=len(importance), evaluated_checkpoint="final_epoch_2000.weights.h5"))
    update(status="COMPLETE", final_weight_evaluation_used=True,
           final_weights_sha256=file_hash(OUTPUT / "final_epoch_2000.weights.h5"))


def run():
    try:
        main()
    except BaseException as error:
        update(status="ERROR", error_type=type(error).__name__, error=str(error))
        (OUTPUT / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    run()

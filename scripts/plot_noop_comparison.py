"""Scientific training curves from saved Keras histories or live API telemetry."""
from __future__ import annotations

import json
from pathlib import Path


def telemetry_history(path):
    rows = {}
    latest_fit = 0
    path = Path(path)
    if not path.exists(): return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            fit = int(row.get("fit", 1))
            if fit > latest_fit:
                rows, latest_fit = {}, fit
            if fit != latest_fit:
                continue
            rows[int(row["epoch"])] = row
        except (ValueError, KeyError, TypeError):
            continue
    keys = sorted(set().union(*(set(r) for r in rows.values())) - {"epoch", "variant", "duration_seconds", "fit"})
    if not rows:
        return {}
    history = {k: [rows[n].get(k, float("nan")) for n in sorted(rows)] for k in keys}
    history["epoch"] = sorted(rows)
    history["fit"] = [latest_fit] * len(rows)
    return history


def metric(history, kind, validation=False):
    prefix = "val_" if validation else ""
    for name in [prefix + kind, prefix + "predictions_" + kind,
                 prefix + "sparse_categorical_" + kind]:
        if name in history: return history[name]
    return []


def save_run_curves(history, path, title="CMI training"):
    return save_comparison_curves({title: history}, path, title=title)


def save_comparison_curves(histories, path, title="Candidate21 baseline and NOOP"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    colors = ["#2265a5", "#c06a24", "#2e8053"]
    for i, (name, history) in enumerate(histories.items()):
        for ax, kind in zip(axes, ("accuracy", "loss")):
            for validation in (False, True):
                values = metric(history, kind, validation)
                if values:
                    epoch_numbers = history.get("epoch", range(1, len(values)+1))
                    if len(epoch_numbers) != len(values):
                        raise ValueError("Epoch numbers and observed metrics must align.")
                    ax.plot(epoch_numbers, values, color=colors[i % len(colors)],
                            linestyle="--" if validation else "-", linewidth=1,
                            label=name + (" validation" if validation else " training"))
            ax.set(xlabel="Epoch", ylabel="Accuracy" if kind == "accuracy" else "Loss")
            ax.grid(alpha=.2)
            if kind == "accuracy": ax.yaxis.set_major_formatter(PercentFormatter(1))
            if ax.lines: ax.legend(fontsize=8)
    fig.suptitle(title + "\nObserved epochs only; dashed lines show internal validation")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return str(path.resolve())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    histories = {}
    for variant in ("baseline", "enhanced"):
        folder = args.directory / variant
        history_path = folder / "history.json"
        history = (json.loads(history_path.read_text()) if history_path.exists()
                   else telemetry_history(folder / "epoch_telemetry.jsonl"))
        if history:
            phase = f" fit {history['fit'][-1]}" if history.get("fit") else " refit"
            histories[variant + phase] = history
    if histories:
        print(save_comparison_curves(histories, args.directory / "accuracy_loss_comparison.png",
              title="Notebook baseline and NOOP: current fits (1 = CV, 2 = refit)"))

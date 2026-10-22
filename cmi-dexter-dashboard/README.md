# CMI Dexter dashboard

A dark Streamlit dashboard using your existing Kaggle API login and CMI Python source.

Run `start.ps1` in PowerShell, then open http://localhost:8501.

- Refresh real status, candidate results and bounded live epoch logs through Kaggle's API.
- Inspect candidate-specific training and validation accuracy/loss curves.
- Compare saved snapshots, inspect full parameter dictionaries and download results.
- Prepare isolated GPU notebooks using your source modules; launch after the active run finishes.
- Edit the procedural interface in `app.py`; reusable processing is in `dashboard_utils.py`.

The app reads your repository without rewriting it. Data snapshots and prepared builds are stored inside this dashboard folder. Notebook submissions preserve the existing Kaggle attachments and sharing settings.

The default server listens on this computer only. Phone access needs a reachable deployment or an explicitly configured LAN listener.

Run checks with `python -m unittest test_dashboard -v`.

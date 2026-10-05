# Local development and Kaggle GPU execution

Use branch `multibranch/v0/develop`, notebook
`notebooks/multibranch_attention.ipynb`, and the modules in `src/`.
The workflow packages the saved notebook and module bytes, adding only a first
bootstrap cell. It does not change architecture, search candidates, data filters,
splits, or the 37 Bayesian iterations.

## Windows setup

The files from `cmi_kaggle_setup.zip` are installed in this repository and the
ignore rules are applied. Do not reapply the archive patch over these files.

Kaggle 2.2.4 requires **Python 3.11 or newer**. Use a separate submission environment
rather than upgrading the existing Python 3.9/3.10 ML environments:

```powershell
Set-Location 'C:\Users\maran\OneDrive\Documents\Git Profile\cmi_dexter'
# Use any existing Python 3.11+ interpreter to create the environment:
& C:\Users\maran\anaconda3\envs\insta_scrape\python.exe -m venv .venv-kaggle
& .venv-kaggle\Scripts\python.exe -m pip install -r requirements-kaggle.txt
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py auth
```

The VS Code tasks explicitly use `.venv-kaggle/Scripts/python.exe`; choosing the
ML notebook interpreter no longer breaks the submission tasks.

Authenticate as `keithmarange` through the browser. Credentials stay outside Git.
On 5 October 2026 the old `~/.kaggle/kaggle.json` key caused permission errors even
after a successful browser login: the CLI preferred that key over OAuth. The old
file was preserved as `~/.kaggle/kaggle.legacy-backup.json`, allowing OAuth to work.
Do not restore it as `kaggle.json` unless that key has been renewed and verified.

The account must have accepted the CMI competition rules and have GPU quota.

## Run and collect results

```powershell
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py prepare
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py push
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py status
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py logs
# Poll every minute and download automatically after COMPLETE:
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py wait
# Or download once after completion:
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py download
```

`push` always rebuilds from the saved local files. Avoid concurrent submissions
from other machines to the same kernel. `prepare` rejects the wrong Git branch.

The generated notebook lives in `kaggle_build/multibranch_attention/`. Its first
cell reconstructs `src/*.py` in `/kaggle/working/src`, installs only missing
scikit-optimize/PyWavelets, checks TensorFlow GPU availability, locates competition
data (including the newer `/kaggle/input/competitions/` layout), and routes outputs
under `/kaggle/working`. Kaggle's TensorFlow installation remains in place.

`submission.json` records the most recently prepared snapshot. After a successful
push, `submitted.json` separately records the acknowledged Kaggle version and
submitted source hash, so a later preparation cannot erase the run association.
The downloader checks COMPLETE and pulls the completed remote source to verify
it matches the acknowledged submission before collecting results. Kaggle 2.2.4's
status/output endpoints use the latest run even if passed a version suffix, so
source verification prevents older or concurrent results being mislabeled.

Downloads go to `results/kaggle/<download-time>/<original-output-path>/`:

- `multibranch_attention_cv.csv`
- `multibranch_attention_best.csv` (best parameters and scores)
- `multibranch_attention_holdout.csv`

The downloader requires a complete set, keeps the execution log and submission
manifest, and avoids downloading the large H5 model. Results and generated builds
are ignored by Git. To inspect a remote copy without overwriting local work:

```powershell
& .venv-kaggle\Scripts\python.exe scripts/kaggle_experiment.py pull
```

## Verification on 5 October 2026

Version **33** was accepted by Kaggle for `keithmarange/multibranch-attention`.
Live streamed logs confirmed two Tesla T4 GPUs, successful local module imports,
CMI data loading (8,151 sequences; 5,113 after filters), the existing 3,856/629
train/holdout split, validation of 90 search parameters, and the first search fit.
Version 33 was subsequently canceled at the user's request. Version 34 was
submitted as a 2-iteration validation run. Its verified completion and all three
CSV downloads are required before restarting the original 37-iteration run.
A five-minute chat follow-up monitors both stages.
No architecture or experiment source files were changed.

Eight workflow tests pass, covering exact experiment/module packaging, acknowledged
version persistence, completed remote source verification, automatic collection,
nested outputs, incomplete outputs and unsafe archives. Run them with:

```powershell
& .venv-kaggle\Scripts\python.exe -m unittest discover -s tests -p test_kaggle_workflow.py -v
```

The broader suite was exercised in `ds_kaggle` with TensorFlow available: 30 tests
ran, with two existing behavior/expectation failures in gradient clipping and
plateau callback construction. These were left unchanged to preserve ML behavior.

## Short validation run

Use push --iterations 2 to change only the packaged Bayesian iteration count.
The local notebook and all other experiment parameters remain unchanged. The
acknowledged manifest records search_iterations and iteration_override.
Submit the original full experiment with push after validating the short run.

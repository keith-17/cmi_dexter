import inspect
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List

def find_data_root(local_dir: str = 'data', kaggle_root: str = '/kaggle/input') -> Path:
    """
    Robustly find the data root directory by searching upwards from the 
    current working directory and the calling script's directory.
    """
    # We only strictly require train.csv to verify the folder. 
    # (test.csv might not be downloaded yet, or demographics might be missing in some setups)
    required_files = ['train.csv'] 
    
    def check_dir(d: Path):
        target = d / local_dir
        if target.exists() and target.is_dir():
            if all((target / f).exists() for f in required_files):
                return target
        return None

    # 1. Check exact path provided (e.g., if user passes an absolute path)
    exact_path = Path(local_dir)
    if exact_path.exists() and exact_path.is_dir():
        if all((exact_path / f).exists() for f in required_files):
            print(f'✅ Using exact path: {exact_path.resolve()}')
            return exact_path

    # Gather starting points for upward search
    search_starts = []
    
    # Current working directory (handles Jupyter/VSCode running from notebooks/)
    search_starts.append(Path.cwd())
    
    # Directory of this file (src/data_utils.py)
    try:
        search_starts.append(Path(__file__).resolve().parent)
    except Exception:
        pass
        
    # Directory of the script/notebook calling this function
    try:
        frame = inspect.currentframe()
        while frame:
            fname = frame.f_globals.get('__file__')
            if fname:
                search_starts.append(Path(fname).resolve().parent)
            frame = frame.f_back
    except Exception:
        pass

    # Remove duplicates while preserving order
    unique_starts = []
    for p in search_starts:
        if p not in unique_starts:
            unique_starts.append(p)

    # 2. Search upwards from all starting points to the root of the filesystem
    for start_path in unique_starts:
        current = start_path.resolve()
        for parent in [current] + list(current.parents):
            res = check_dir(parent)
            if res:
                print(f'✅ Found data folder: {res}')
                return res

    # 3. Check Kaggle environment
    kaggle_path = Path(kaggle_root)
    if kaggle_path.exists():
        for candidate in kaggle_path.rglob('train.csv'):
            target_dir = candidate.parent
            print(f'✅ Using Kaggle data folder: {target_dir}')
            return target_dir

    # 4. Fallback / Error
    raise FileNotFoundError(
        f'❌ Could not find the dataset.\n'
        f'Searched upwards from: {[str(p) for p in unique_starts]}\n'
        f'Looked for a folder named "{local_dir}" containing {required_files}.\n'
        f'Current directory: {Path.cwd()}\n'
        f'Please ensure your CSV files are in a "data/" folder at your project root.'
    )


def sample_balanced_split(
    df: pd.DataFrame,
    train_pct: float = 0.20,
    test_pct: float = 0.05,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split sequences into train and test sets balanced by subject and gesture.
    """
    total_sequences = df['sequence_id'].nunique()
    n_gestures = df['gesture'].nunique()
    n_subjects = df['subject'].nunique()

    train_target = int(total_sequences * train_pct)
    test_target = int(total_sequences * test_pct)

    train_seqs_per_cell = train_target // (n_subjects * n_gestures)
    test_seqs_per_cell = test_target // (n_subjects * n_gestures)

    min_pct = n_subjects * n_gestures / total_sequences

    if train_seqs_per_cell == 0 or test_seqs_per_cell == 0:
        raise ValueError(
            f"Percentage too small. Min viable pct for this data: {min_pct:.1%}"
        )

    train_ids = []
    test_ids = []

    for _, group in df.groupby(['subject', 'gesture']):
        unique_seqs = group['sequence_id'].drop_duplicates().sample(frac=1, random_state=random_state)
        
        n_train = min(train_seqs_per_cell, len(unique_seqs))
        n_test = min(test_seqs_per_cell, len(unique_seqs) - n_train)
        
        train_ids.extend(unique_seqs.iloc[:n_train].tolist())
        test_ids.extend(unique_seqs.iloc[n_train:n_train + n_test].tolist())

    train_df = df[df['sequence_id'].isin(train_ids)]
    test_df = df[df['sequence_id'].isin(test_ids)]

    assert len(set(train_ids) & set(test_ids)) == 0, "Overlap detected!"

    print(f"Train: {train_df['sequence_id'].nunique()} seqs | {100 * train_df['sequence_id'].nunique() / total_sequences:.1f}%")
    print(f"Test:  {test_df['sequence_id'].nunique()} seqs  | {100 * test_df['sequence_id'].nunique() / total_sequences:.1f}%")

    return train_df, test_df
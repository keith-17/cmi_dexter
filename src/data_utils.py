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
            print(f'Using exact path: {exact_path.resolve()}')
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
                print(f'Found data folder: {res}')
                return res

    # 3. Check Kaggle environment
    kaggle_path = Path(kaggle_root)
    if kaggle_path.exists():
        for candidate in kaggle_path.rglob('train.csv'):
            target_dir = candidate.parent
            print(f'Using Kaggle data folder: {target_dir}')
            return target_dir

    # 4. Fallback / Error
    raise FileNotFoundError(
        f'Could not find the dataset.\n'
        f'Searched upwards from: {[str(p) for p in unique_starts]}\n'
        f'Looked for a folder named "{local_dir}" containing {required_files}.\n'
        f'Current directory: {Path.cwd()}\n'
        f'Please ensure your CSV files are in a "data/" folder at your project root.'
    )


def _largest_remainder(quotas: np.ndarray, caps: np.ndarray, target: int, rng) -> np.ndarray:
    """Round per-cell quotas to integers that sum to ``target`` without exceeding ``caps``."""
    counts = np.minimum(np.floor(quotas).astype(int), caps)
    remaining = int(target - counts.sum())
    if remaining <= 0:
        return counts
    fractions = quotas - np.floor(quotas)
    shuffled = rng.permutation(len(quotas))
    order = shuffled[np.argsort(-fractions[shuffled], kind='stable')]
    while remaining > 0:
        added = False
        for idx in order:
            if remaining == 0:
                break
            if counts[idx] < caps[idx]:
                counts[idx] += 1
                remaining -= 1
                added = True
        if not added:
            break
    return counts


def sample_balanced_split(
    df: pd.DataFrame,
    train_pct: float = 0.20,
    test_pct: float = 0.05,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split sequences into disjoint train and test sets, stratified by subject and gesture.

    Each (subject, gesture) cell contributes to both sets in proportion to its size, and the
    totals match ``train_pct`` / ``test_pct`` of all sequences. When the two fractions sum
    to 1 every sequence is assigned to exactly one set.
    """
    if train_pct <= 0 or test_pct <= 0 or train_pct + test_pct > 1 + 1e-9:
        raise ValueError('train_pct and test_pct must be positive and sum to at most 1.')

    rng = np.random.default_rng(random_state)
    total_sequences = df['sequence_id'].nunique()
    cells = [
        group['sequence_id'].drop_duplicates().sample(frac=1, random_state=random_state).tolist()
        for _, group in df.groupby(['subject', 'gesture'])
    ]
    sizes = np.array([len(cell) for cell in cells])

    train_target = int(round(total_sequences * train_pct))
    n_train = _largest_remainder(sizes * train_pct, sizes, train_target, rng)

    if train_pct + test_pct >= 1 - 1e-9:
        test_target = total_sequences - int(n_train.sum())
    else:
        test_target = min(int(round(total_sequences * test_pct)), total_sequences - int(n_train.sum()))
    n_test = _largest_remainder(sizes * test_pct, sizes - n_train, test_target, rng)

    train_ids, test_ids = [], []
    for cell, k_train, k_test in zip(cells, n_train, n_test):
        train_ids.extend(cell[:k_train])
        test_ids.extend(cell[k_train:k_train + k_test])

    assert len(set(train_ids) & set(test_ids)) == 0, "Overlap detected!"

    train_df = df[df['sequence_id'].isin(train_ids)]
    test_df = df[df['sequence_id'].isin(test_ids)]

    print(f"Train: {len(train_ids)} seqs | {100 * len(train_ids) / total_sequences:.1f}%")
    print(f"Test:  {len(test_ids)} seqs  | {100 * len(test_ids) / total_sequences:.1f}%")
    unused = total_sequences - len(train_ids) - len(test_ids)
    if unused:
        print(f"Unused: {unused} seqs | {100 * unused / total_sequences:.1f}%")

    return train_df, test_df

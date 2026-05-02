from pathlib import Path

import pandas as pd


def find_data_root(local_dir: str = 'data', kaggle_root: str = '/kaggle/input') -> Path:
    local_path = Path(local_dir)
    required = [
        'train.csv',
        'test.csv',
        'train_demographics.csv',
        'test_demographics.csv'
    ]

    if local_path.exists() and all((local_path / f).exists() for f in required):
        print(f'Using local data folder: {local_path.resolve()}')
        return local_path

    kaggle_root = Path(kaggle_root)
    if kaggle_root.exists():
        for csv_path in kaggle_root.rglob('train.csv'):
            candidate = csv_path.parent
            if all((candidate / f).exists() for f in required):
                print(f'Using Kaggle data folder: {candidate}')
                return candidate

    raise FileNotFoundError(
        'Could not find the dataset locally or in /kaggle/input. '
        'Place the CSV files in ./data/ or attach the Kaggle dataset.'
    )


def sample_balanced_split(df: pd.DataFrame, train_pct: float = 0.20, test_pct: float = 0.05, random_state: int = 42)\
        -> (pd.DataFrame, pd.DataFrame):
    total_sequences = df['sequence_id'].nunique()
    n_gestures = df['gesture'].nunique()
    n_subjects = df['subject'].nunique()

    train_target = int(total_sequences * train_pct)
    test_target  = int(total_sequences * test_pct)

    train_seqs_per_cell = train_target // (n_subjects * n_gestures)
    test_seqs_per_cell  = test_target  // (n_subjects * n_gestures)

    min_pct = n_subjects * n_gestures / total_sequences

    if train_seqs_per_cell == 0 or test_seqs_per_cell == 0:
        raise ValueError(
            f"Percentage too small. "
            f"Min viable pct for this data: {min_pct:.1%}"
        )

    train_ids, test_ids = [], []

    for _, group in df.groupby(['subject', 'gesture']):
        unique_seqs = group['sequence_id'].drop_duplicates().sample(frac=1, random_state=random_state)

        n_train = min(train_seqs_per_cell, len(unique_seqs))
        n_test  = min(test_seqs_per_cell,  len(unique_seqs) - n_train)

        train_ids.extend(unique_seqs.iloc[:n_train].tolist())
        test_ids.extend( unique_seqs.iloc[n_train:n_train + n_test].tolist())

    train_df = df[df['sequence_id'].isin(train_ids)]
    test_df = df[df['sequence_id'].isin(test_ids)]

    assert len(set(train_ids) & set(test_ids)) == 0, "Overlap detected!"

    print(f"Train: {train_df['sequence_id'].nunique()} seqs | {100*train_df['sequence_id'].nunique()/total_sequences:.1f}%")
    print(f"Test:  {test_df['sequence_id'].nunique()} seqs  | {100*test_df['sequence_id'].nunique()/total_sequences:.1f}%")

    return train_df, test_df
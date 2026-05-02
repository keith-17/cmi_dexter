from pathlib import Path


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
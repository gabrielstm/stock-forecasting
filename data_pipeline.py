from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import NormalizeMult, NormalizeMultUseData, create_dataset
import config

# Calculate split index dynamically if possible, but here we need a default.
# We can read the csv to get length, or just use a placeholder if it's always passed.
# However, nlinear and patchTST use these defaults.
# Let's try to read the CSV to set the default correctly, or just import config.
# Since config has a helper, we can use it if we load the data.
# But loading data at module level is bad practice.
# We will update nlinear.py and patchTST.py to pass the correct split index from config.
# Here we can just set defaults to None or keep them as fallback, but updated to use config values where possible.

STOCK_CSV = config.DATASET_NAME
# We can't easily get len(data) here without reading file.
# Let's read it once to set the constant, or better, change prepare_windows to calculate it if not provided.
_temp_df = pd.read_csv(STOCK_CSV)
DATA_SPLIT_INDEX = config.get_split_index(len(_temp_df))
del _temp_df

TIME_STEPS_DEFAULT = config.WINDOW_SIZE
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']
TARGET_COLUMN = 'close'
RESIDUAL_CSV = 'data/ARIMA_residuals1.csv'


def load_merged_frame(
    stock_path: Path,
    residual_path: Path,
    feature_columns: List[str]
) -> pd.DataFrame:
    stock_df = pd.read_csv(stock_path)
    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'], format='%Y%m%d')
    # stock_df = stock_df[['trade_date'] + feature_columns].copy() # Removed early filtering

    residual_df = pd.read_csv(residual_path)
    if 'trade_date' in residual_df.columns:
        residual_df['trade_date'] = pd.to_datetime(residual_df['trade_date'])
    else:
        residual_df.index = pd.to_datetime(residual_df.index)
        residual_df = residual_df.reset_index().rename(columns={'index': 'trade_date'})

    merged = pd.merge(stock_df, residual_df, on='trade_date', how='inner')
    merged = merged.sort_values('trade_date').set_index('trade_date')
    
    # Filter columns after merge to allow features from either stock or residual
    # Ensure all requested feature columns exist
    available_columns = merged.columns.tolist()
    missing_columns = [col for col in feature_columns if col not in available_columns]
    if missing_columns:
        # If '0' is missing, it might be because ARIMA_residuals1.csv hasn't been updated yet or has different name
        # But assuming it works if files are correct.
        # For now, let's just proceed. If '0' is missing, it will fail here, which is better.
        pass

    merged = merged[feature_columns]
    return merged


def prepare_windows(
    time_steps: int = TIME_STEPS_DEFAULT,
    split_index: int = DATA_SPLIT_INDEX,
    target_column: str = TARGET_COLUMN,
    feature_columns: List[str] = FEATURE_COLUMNS,
    stock_csv: str = STOCK_CSV,
    residual_csv: str = RESIDUAL_CSV
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, pd.DatetimeIndex]:
    merged = load_merged_frame(Path(stock_csv), Path(residual_csv), feature_columns)
    feature_names = merged.columns.tolist()
    if target_column not in feature_names:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada nas colunas disponíveis: {feature_names}")

    target_idx = feature_names.index(target_column)

    train_df = merged.iloc[1:split_index].copy()
    test_df = merged.iloc[split_index:].copy()

    train_array, normalize = NormalizeMult(train_df.values)
    test_array = NormalizeMultUseData(test_df.values.copy(), normalize.copy())

    train_X, train_Y = create_dataset(train_array, time_steps)
    test_X, test_Y = create_dataset(test_array, time_steps)

    train_y = train_Y[:, target_idx].astype(np.float32)
    test_y = test_Y[:, target_idx].astype(np.float32)

    train_X = train_X.astype(np.float32)
    test_X = test_X.astype(np.float32)

    test_dates = test_df.index[time_steps + 1:]

    return train_X, train_y, test_X, test_y, normalize, target_idx, test_dates


def inverse_scale(values: np.ndarray, normalize: np.ndarray, target_idx: int) -> np.ndarray:
    median, iqr = normalize[target_idx]
    return values * iqr + median

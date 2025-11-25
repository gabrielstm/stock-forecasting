from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import NormalizeMult, NormalizeMultUseData, create_dataset

DATA_SPLIT_INDEX = 3500
TIME_STEPS_DEFAULT = 20
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'vol', 'amount']
TARGET_COLUMN = 'close'
STOCK_CSV = '601988.SH.csv'
RESIDUAL_CSV = 'ARIMA_residuals1.csv'


def load_merged_frame(
    stock_path: Path,
    residual_path: Path,
    feature_columns: List[str]
) -> pd.DataFrame:
    stock_df = pd.read_csv(stock_path)
    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'], format='%Y%m%d')
    stock_df = stock_df[['trade_date'] + feature_columns].copy()

    residual_df = pd.read_csv(residual_path)
    if 'trade_date' in residual_df.columns:
        residual_df['trade_date'] = pd.to_datetime(residual_df['trade_date'])
    else:
        residual_df.index = pd.to_datetime(residual_df.index)
        residual_df = residual_df.reset_index().rename(columns={'index': 'trade_date'})

    merged = pd.merge(stock_df, residual_df, on='trade_date', how='inner')
    merged = merged.sort_values('trade_date').set_index('trade_date')
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
    low, high = normalize[target_idx]
    return values * (high - low) + low

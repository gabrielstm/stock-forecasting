import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras import layers, models, optimizers

from data_pipeline import (DATA_SPLIT_INDEX, FEATURE_COLUMNS, RESIDUAL_CSV,
                           STOCK_CSV, TARGET_COLUMN, TIME_STEPS_DEFAULT,
                           inverse_scale, prepare_windows)
from utils import evaluation_metric

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

def build_model(input_shape):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ]
    )
    return model


def plot_results(dates, y_true, y_pred, title: str, save_name: str, show_plot: bool):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Real')
    plt.plot(dates, y_pred, label='Predição')
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.legend()
    plt.tight_layout()
    output_path = RESULTS_DIR / save_name
    plt.savefig(output_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_test_results(model_name: str, dates, y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = mse**0.5
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    df = pd.DataFrame({
        'date': dates,
        'true': y_true,
        'pred': y_pred,
        'abs_error': np.abs(y_true - y_pred)
    })

    results_path = RESULTS_DIR / f'{model_name}_test_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f'MSE: {mse:.6f}\n')
        f.write(f'RMSE: {rmse:.6f}\n')
        f.write(f'MAE: {mae:.6f}\n')
        f.write(f'R2: {r2:.6f}\n')
        f.write('\n')
        df.to_csv(f, index=False)


def train(args):
    train_X, train_y, test_X, test_y, normalize, target_idx, test_dates = prepare_windows(
        args.time_steps, args.split_index
    )

    model = build_model((args.time_steps, train_X.shape[2]))
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    history = model.fit(
        train_X,
        train_y,
        validation_data=(test_X, test_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    preds = model.predict(test_X, batch_size=args.batch_size).flatten()

    y_true = inverse_scale(test_y, normalize, target_idx)
    y_pred = inverse_scale(preds, normalize, target_idx)

    evaluation_metric(y_true, y_pred)

    dates = test_dates[: len(y_true)]
    plot_results(dates, y_true, y_pred, 'NLinear vs Real (Close)', 'nlinear_predictions.png', not args.no_plot)
    save_test_results('nlinear', dates, y_true, y_pred)

    return history


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline NLinear model for stock forecasting.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--time-steps', type=int, default=TIME_STEPS_DEFAULT)
    parser.add_argument('--split-index', type=int, default=DATA_SPLIT_INDEX)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--no-plot', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()

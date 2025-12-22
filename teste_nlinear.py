import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras import layers, models, optimizers, callbacks

from data_pipeline_indicadores import (DATA_SPLIT_INDEX, FEATURE_COLUMNS, RESIDUAL_CSV,
                           STOCK_CSV, TARGET_COLUMN, TIME_STEPS_DEFAULT,
                           inverse_scale, prepare_windows)
from utils import evaluation_metric

import config

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

def build_model(input_shape, target_idx=0, use_regularization=True):
    """
    NLinear model implementation for time series forecasting.
    
    The NLinear architecture:
    1. Subtracts the last timestep value to handle non-stationarity
    2. Applies a linear transformation
    3. Adds back the last timestep value for the final prediction
    
    This simple approach often outperforms complex models in many scenarios.
    """
    inputs = layers.Input(shape=input_shape)
    
    # NLinear: subtract last value to handle non-stationarity
    seq_last = layers.Lambda(lambda x: x[:, -1:, :])(inputs)
    x = layers.Subtract()([inputs, seq_last])
    
    # Flatten the input
    x = layers.Flatten()(x)
    
    # Linear layer with optional L2 regularization to prevent overfitting
    if use_regularization:
        x = layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    else:
        x = layers.Dense(1)(x)
    
    # Add back the last value of the target feature
    target_last = layers.Lambda(lambda x: x[:, -1:, target_idx])(inputs)
    target_last = layers.Reshape((1,))(target_last)
    
    outputs = layers.Add()([x, target_last])
    
    model = models.Model(inputs=inputs, outputs=outputs)
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


def plot_loss(history, save_name: str, show_plot: bool):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Treino (Loss)')
    plt.plot(history.history['val_loss'], label='Validação (Loss)')
    plt.title('Curva de Perda do Modelo (MSE)')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)
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
    # Use residuals as features and target
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'RSI', 'EMA12', 'ATR', 'KC_Width', 'OBV', 'BB_Width', 'CCI']
    #feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    target_column = 'close'

    train_X, train_y, test_X, test_y, normalize, target_idx, test_dates = prepare_windows(
        args.time_steps, args.split_index, target_column=target_column, feature_columns=feature_columns
    )

    # Split training data into train and validation sets (80/20 split)
    # CRITICAL: This prevents look-ahead bias by NOT using test data during training
    val_split = int(len(train_X) * 0.8)
    
    X_train = train_X[:val_split]
    y_train = train_y[:val_split]
    X_val = train_X[val_split:]
    y_val = train_y[val_split:]
    
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(test_X)}")

    model = build_model((args.time_steps, train_X.shape[2]), target_idx=target_idx, 
                       use_regularization=args.use_regularization)
    
    # Reduced learning rate for better stability
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Callbacks for better training
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=args.patience // 2,
        min_lr=1e-7,
        verbose=1
    )
    
    model_callbacks = [early_stop, reduce_lr]

    # Train with proper validation set (NOT test set)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=model_callbacks,
        verbose=1
    )

    preds = model.predict(test_X, batch_size=args.batch_size).flatten()

    y_true_final = inverse_scale(test_y, normalize, target_idx)
    y_pred_final = inverse_scale(preds, normalize, target_idx)

    # Ajuste das datas para o gráfico
    dates = test_dates[: len(y_pred_final)]
    
    # 3. Avaliação e Plotagem Direta
    print("Evaluation on Direct Price Prediction (Close):")
    evaluation_metric(y_true_final, y_pred_final)

    # Plotagem sem a necessidade de somar nada
    plot_results(dates, y_true_final, y_pred_final, 
                'NLinear Direct Prediction vs Real', 'nlinear_direct_predictions.png', not args.no_plot)
    
    save_test_results('nlinear_direct', dates, y_true_final, y_pred_final)
    plot_loss(history, 'nlinear_direct_loss_curve.png', not args.no_plot)

    return history


def parse_args():
    parser = argparse.ArgumentParser(description='NLinear model for stock trading forecasting.')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--time-steps', type=int, default=TIME_STEPS_DEFAULT,
                       help='Number of time steps in input sequence')
    parser.add_argument('--split-index', type=int, default=DATA_SPLIT_INDEX,
                       help='Index to split train/test data')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Initial learning rate (reduced for stability)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--use-regularization', action='store_true', default=True,
                       help='Use L2 regularization')
    parser.add_argument('--no-regularization', dest='use_regularization', 
                       action='store_false',
                       help='Disable L2 regularization')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plot display')
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()

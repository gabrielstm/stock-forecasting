import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras import layers, models, optimizers

from data_pipeline_indicadores import (DATA_SPLIT_INDEX, FEATURE_COLUMNS, RESIDUAL_CSV,
                           STOCK_CSV, TARGET_COLUMN, TIME_STEPS_DEFAULT,
                           inverse_scale, prepare_windows)
from utils import evaluation_metric
import config

PATCH_LEN_DEFAULT = 4
PATCH_STRIDE_DEFAULT = 2
D_MODEL_DEFAULT = 64
NUM_HEADS_DEFAULT = 4
FF_DIM_DEFAULT = 128
NUM_ENCODER_LAYERS = 2
DROPOUT_RATE = 0.1
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def compute_num_patches(time_steps: int, patch_len: int, stride: int) -> int:
    if patch_len > time_steps:
        raise ValueError('patch_len deve ser menor ou igual a time_steps')
    return math.floor((time_steps - patch_len) / stride) + 1


def transformer_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float):
    # Multi-head attention (self-attention)
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed-forward network
    ffn = layers.Dense(ff_dim, activation='gelu')(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(d_model)(ffn)

    return layers.LayerNormalization(epsilon=1e-6)(x + ffn)


def build_patchtst_model(
    time_steps: int,
    num_features: int,
    patch_len: int,
    stride: int,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    num_layers: int,
    dropout: float
) -> models.Model:
    """
    Implementação fiel ao comportamento do PatchTST:
    - extrai patches por feature (channel-independent),
    - projeta cada patch_len -> d_model por feature,
    - cria tokens (num_patches * num_features),
    - adiciona positional embeddings por patch e embedding por canal,
    - aplica encoder Transformer e cabeça final.
    """

    num_patches = compute_num_patches(time_steps, patch_len, stride)

    inputs = layers.Input(shape=(time_steps, num_features), name='inputs')

    # 1) Extrair patches ao longo do eixo temporal:
    # tf.signal.frame: (batch, num_patches, patch_len, num_features)
    frames = layers.Lambda(
        lambda x: tf.signal.frame(x, frame_length=patch_len, frame_step=stride, axis=1),
        name='extract_patches'
    )(inputs)

    # 2) Rearranjar para (batch, num_patches, num_features, patch_len)
    frames = layers.Permute((1, 3, 2), name='permute_patches')(frames)

    # 3) Projetar cada patch (por feature) de length patch_len -> d_model
    #    Usamos TimeDistributed(TimeDistributed(Dense)) para aplicar Dense ao último eixo patch_len
    proj_per_feature = layers.TimeDistributed(
        layers.TimeDistributed(layers.Dense(d_model), name='patch_proj_dense'),
        name='td_td_patch_proj'
    )(frames)  # (batch, num_patches, num_features, d_model)

    # 4) Positional embeddings:
    # - embedding para posição de patch (num_patches)
    # - embedding para canal/feature (num_features)
    patch_pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=d_model, name='patch_pos_emb')
    chan_embedding = layers.Embedding(input_dim=num_features, output_dim=d_model, name='chan_emb')

    def add_pos_chan(x):
        # x: (batch, num_patches, num_features, d_model)
        b = tf.shape(x)[0]
        # patch positions: (num_patches, d_model) -> (1, num_patches, 1, d_model) -> tiled
        patch_idx = tf.range(num_patches)
        p_emb = patch_pos_embedding(patch_idx)  # (num_patches, d_model)
        p_emb = tf.reshape(p_emb, (1, num_patches, 1, d_model))
        p_emb = tf.tile(p_emb, (b, 1, num_features, 1))

        # channel embeddings: (num_features, d_model) -> (1, 1, num_features, d_model) -> tiled
        chan_idx = tf.range(num_features)
        c_emb = chan_embedding(chan_idx)  # (num_features, d_model)
        c_emb = tf.reshape(c_emb, (1, 1, num_features, d_model))
        c_emb = tf.tile(c_emb, (b, num_patches, 1, 1))

        return x + p_emb + c_emb

    x = layers.Lambda(add_pos_chan, name='add_pos_chan')(proj_per_feature)  # same shape

    # 5) Transformar para tokens: (batch, num_patches * num_features, d_model)
    seq_len = num_patches * num_features
    x = layers.Reshape((seq_len, d_model), name='to_tokens')(x)

    # 6) Encoder: aplicar num_layers blocos Transformer
    for i in range(num_layers):
        x = transformer_block(x, d_model, num_heads, ff_dim, dropout)

    # 7) Cabeça final: flatten + MLP + saída de 1 valor (regressão)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(dropout, name='dropout_head')(x)
    x = layers.Dense(ff_dim, activation='relu', name='dense_head')(x)
    outputs = layers.Dense(1, name='regression_out')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='PatchTST_TF')


def plot_loss(history, save_name: str, show_plot: bool):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Treino (Loss)')
    plt.plot(history.history['val_loss'], label='Validação (Loss)')
    plt.title('PatchTST: Curva de Perda do Modelo (MSE)')
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


def plot_results(dates, y_true, y_pred, save_name: str, show_plot: bool):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Real')
    plt.plot(dates, y_pred, label='PatchTST')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento')
    plt.title('PatchTST vs Real (Close)')
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
    # Use residuals as features and target
    feature_columns = ['0']
    #feature_columns = ['0', 'open', 'high', 'low', 'close', 'volume', 'amount', 'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'RSI', 'EMA12', 'ATR', 'KC_Width', 'OBV', 'BB_Width', 'CCI']
    #feature_columns = ['0', 'open', 'high', 'low', 'close', 'volume', 'amount']
    target_column = '0'

    train_X, train_y, test_X, test_y, normalize, target_idx, test_dates = prepare_windows(
        time_steps=args.time_steps,
        split_index=args.split_index,
        target_column=target_column,
        feature_columns=feature_columns,
        stock_csv=STOCK_CSV,
        residual_csv=RESIDUAL_CSV
    )

    model = build_patchtst_model(
        time_steps=args.time_steps,
        num_features=train_X.shape[2],
        patch_len=args.patch_len,
        stride=args.patch_stride,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.layers,
        dropout=args.dropout
    )

    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    model.summary()

    history = model.fit(
        train_X,
        train_y,
        validation_data=(test_X, test_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    print("Gerando gráfico de perda para PatchTST...")
    plot_loss(history, 'patchtst_loss_curve.png', not args.no_plot)

    preds = model.predict(test_X, batch_size=args.batch_size).flatten()

    y_true_residual = inverse_scale(test_y, normalize, target_idx)
    y_pred_residual = inverse_scale(preds, normalize, target_idx)
    
    # Load ARIMA predictions
    arima_preds = pd.read_csv('data/ARIMA.csv')
    arima_preds['trade_date'] = pd.to_datetime(arima_preds['trade_date'])
    arima_preds = arima_preds.set_index('trade_date')

    # Create DataFrame for predicted residuals
    dates = test_dates[: len(y_pred_residual)]
    pred_residuals_df = pd.DataFrame({
        'trade_date': dates,
        'residual_pred': y_pred_residual
    }).set_index('trade_date')

    # Combine ARIMA + Residuals
    arima_preds = arima_preds.rename(columns={'close': 'close'})
    pred_residuals_df = pred_residuals_df.rename(columns={'residual_pred': 'close'})
    
    final_pred = pd.concat([arima_preds, pred_residuals_df]).groupby('trade_date')['close'].sum()
    
    # Load Real Stock Price for comparison
    stock_df = pd.read_csv(STOCK_CSV)
    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'], format='%Y%m%d')
    stock_df = stock_df.set_index('trade_date')
    
    # Align with final_pred
    test_dates_intersection = final_pred.index.intersection(stock_df.index).intersection(dates)
    
    final_pred = final_pred.loc[test_dates_intersection]
    y_true_final = stock_df.loc[test_dates_intersection, 'close']
    
    print("Evaluation on Final Prediction (ARIMA + Residuals):")
    evaluation_metric(y_true_final.values, final_pred.values)

    plot_results(test_dates_intersection, y_true_final.values, final_pred.values, 'patchtst_predictions.png', not args.no_plot)
    save_test_results('patchtst', test_dates_intersection, y_true_final.values, final_pred.values)

    return history


def parse_args():
    parser = argparse.ArgumentParser(description='PatchTST implementation em TensorFlow (corrigida).')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--time-steps', type=int, default=TIME_STEPS_DEFAULT)
    parser.add_argument('--split-index', type=int, default=DATA_SPLIT_INDEX)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--patch-len', type=int, default=PATCH_LEN_DEFAULT)
    parser.add_argument('--patch-stride', type=int, default=PATCH_STRIDE_DEFAULT)
    parser.add_argument('--d-model', type=int, default=D_MODEL_DEFAULT)
    parser.add_argument('--num-heads', type=int, default=NUM_HEADS_DEFAULT)
    parser.add_argument('--ff-dim', type=int, default=FF_DIM_DEFAULT)
    parser.add_argument('--layers', type=int, default=NUM_ENCODER_LAYERS)
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE)
    parser.add_argument('--no-plot', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()

import pandas as pd
import pandas_ta as ta
import numpy as np

def add_technical_indicators(df_b3: pd.DataFrame) -> pd.DataFrame:

    
    df = df_b3.copy()

    # MACD (Moving Average Convergence Divergence) - Momento/Tendência
    macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9, append=False)
    df['MACD_Line'] = macd_data['MACD_12_26_9']
    df['MACD_Signal'] = macd_data['MACDs_12_26_9']
    df['MACD_Hist'] = macd_data['MACDh_12_26_9']

    # RSI (Relative Strength Index) - Momento
    df['RSI'] = ta.rsi(df['close'], length=14, append=False)

    # EMA (Exponential Moving Average) - Tendência
    df['EMA12'] = ta.ema(df['close'], length=12)

    # ATR (Average True Range) - Volatilidade
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14, append=False)

    # Keltner Channels (KC) - Volatilidade/Tendência (Usamos apenas a largura/Amplitude)
    kc_data = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2, append=False)
    df['KC_Width'] = kc_data['KCw_20_2.0']

    # On-Balance Volume (OBV) - Volume
    df['OBV'] = ta.obv(df['close'], df['volume'], append=False)

    # Bandas de Bollinger (BBands) - Volatilidade
    bbands_data = ta.bbands(df['close'], length=20, std=2, append=False)
    df['BB_Width'] = bbands_data['BBLw_20_2.0']

    # Commodity Channel Index (CCI) - Momento Extremo
    df['CCI'] = ta.cci(df['high'], df['low'], df['close'], length=14, append=False)


    # Indicadores técnicos geralmente têm valores NaN no início da série 
    # Precisamos preencher esses NaNs para o modelo DL.
    # Utilizamos o valor anterior válido (ffill) e, em seguida, com 0 para aqueles que não tiver
    # (primeiros NaNs).
    df = df.fillna(method='ffill').fillna(0)
    
    return df
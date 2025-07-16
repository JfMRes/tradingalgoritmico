import pandas as pd
from functions import *
from visualizacion import *
from train import *


moneda = 'BTCUSDT'


# Carga de BTC
df_btc = read_data(moneda)


df_btc, col_rsi = add_rsi(df_btc)
df_btc, col_outcome, col_gain_bool = add_trade_outcome(df_btc, horizon=24, take_profit=3, stop_loss=1)
df_btc, col_ema = add_ema(df_btc, period=12, price_col='close', verbose=True)
df_btc, col_ema_cross = add_ema_cross(df_btc, fast=12, slow=26, price_col='close', verbose=True)

# Filtramos solo por las ultimas fechas
df_btc = filtrar_fecha(df_btc, anios = 5)

ranking = calcular_importancia_features(df_btc)

df_btc, _ = balanced_methods(df_btc)

execute_random_forest(df_btc)


# save_checkpoint(df_btc, moneda)
# print(df_btc.tail())
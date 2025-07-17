import pandas as pd
from functions import *
from visualizacion import *
from train import *
from inferencia import predict_from_model
from config import *
from backtesting import *

moneda = 'BTCUSDT'


# Carga de BTC
df_btc = read_data(moneda)


df_btc, col_rsi = add_rsi(df_btc)
df_btc, col_ema = add_ema(df_btc, period=12, price_col='close', verbose=True)
df_btc, col_ema_cross = add_ema_cross(df_btc, fast=12, slow=26, price_col='close', verbose=True)

# Filtramos solo por las ultimas fechas
df_btc, df_test = filtrar_fecha(df_btc, total_anios = 5, eliminar_anios_final = 1)

# Obtenemos las columnas de resultados para cada caso
df_btc, col_outcome, col_gain_bool = add_trade_outcome(df_btc, horizon=24, take_profit = TAKE_PROFIT, stop_loss= STOP_LOSS)

# Obtenemos la dependencia con la salida
ranking = calcular_importancia_features(df_btc)

# Balanceamos datos
df_btc, _ = balanced_methods(df_btc)

# Creamos el modelo
modelo, feature_cols, target_col = execute_random_forest(df_btc, )

# Predecimos con el modelo ya entrenado para una fila o varias
pred = predict_from_model(df_test, modelo, feature_cols, threshold=THRESHOLD, return_probs=True)

pred.to_csv('predicciones.csv', index=False)

back = backtesting(pred)

back_testing_resume(back)
back_testing_graph(back)

back.to_csv('back.csv')

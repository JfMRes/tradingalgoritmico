import pandas as pd

def read_data(cambio: str):
    df = pd.read_csv(f'data/{cambio}.csv')

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S+00:00')

    # Ordenamos por fecha de menos a mayor
    df.sort_values(by='date', inplace=True)

    return df

def save_checkpoint(df, moneda):
    """
    Guarda el DataFrame en un archivo CSV con el nombre de la moneda _ el nombre de todas las columnas excepto las 6 primeras.
    """

    nombre_columnas = df.columns[6:].tolist()
    if not nombre_columnas:
        print("⚠️ No hay columnas para guardar.")
        return
    nombre_archivo = f'checkpoints/{moneda}_{"_".join(nombre_columnas)}.csv'
    df.to_csv(nombre_archivo, index=False)
    print(f"✅ Checkpoint guardado como {nombre_archivo}")

def filtrar_fecha(df, total_anios=5, eliminar_anios_final=1):
    if 'date' not in df.columns:
        raise ValueError("El DataFrame debe tener una columna 'date'")
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    fecha_max = df['date'].max()
    fecha_inicio_total = fecha_max - pd.DateOffset(years=total_anios)
    fecha_inicio_valido = fecha_max - pd.DateOffset(years=eliminar_anios_final)
    
    df_filtrado = df[(df['date'] >= fecha_inicio_total) & (df['date'] < fecha_inicio_valido)]
    df_eliminado = df[df['date'] >= fecha_inicio_valido]
    
    return df_filtrado, df_eliminado

def add_rsi(df, period=14, verbose=True):

    """
    Añade una columna RSI al DataFrame usando el precio de cierre.

    Args:
        df (pd.DataFrame): DataFrame base con columnas 'close'.
        period (int): Número de velas pasadas que se usan para calcular el RSI. 
                      Valores típicos: 14 (clásico), 21, etc.
        verbose (bool): Si True, imprime una descripción del indicador al generarlo.

    Returns:
        pd.DataFrame: DataFrame original con una nueva columna 'rsi_{period}'.
        col_name (str): Nombre de la nueva columna RSI.

    Indicador:
        RSI (Relative Strength Index) mide la fuerza del movimiento del precio.
        Valores altos indican sobrecompra (>70), valores bajos indican sobreventa (<30).
    """

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    col_name = f'rsi_{period}'
    df[col_name] = rsi

    if verbose:
        print(f"✅ Añadido {col_name}: mide momentum (fuerza relativa) en los últimos {period} cierres. "
              f"Valores >70 = sobrecompra, <30 = sobreventa.")

    return df, col_name

def add_ema(df, period=12, price_col='close', verbose=True):
    """
    Añade una columna EMA (media móvil exponencial) al DataFrame.

    Args:
        df (pd.DataFrame): DataFrame base con columna de precio.
        period (int): Número de velas pasadas para calcular la EMA. 
                      Común usar 12, 26, 50, 100, 200 según el horizonte.
        price_col (str): Columna de precio sobre la que se calcula (por defecto 'close').
        verbose (bool): Si True, imprime descripción del indicador generado.

    Returns:
        pd.DataFrame: DataFrame con nueva columna 'ema_{period}'.
        col_name (str): Nombre de la columna creada.

    Indicador:
        EMA da más peso a precios recientes que una media móvil simple.
        Útil para detectar tendencias con menos retraso.
        Cruces de EMAs (ej: EMA12 y EMA26) generan señales clásicas de entrada/salida.
    """
    col_name = f'ema_{period}'
    df[col_name] = df[price_col].ewm(span=period, adjust=False).mean()

    if verbose:
        print(f"✅ Añadido {col_name}: media móvil exponencial sobre {price_col} en {period} velas. "
              f"Señala tendencias con mayor sensibilidad que una SMA.")

    return df, col_name

def add_ema_cross(df, fast=12, slow=26, price_col='close', verbose=True):
    """
    Añade columnas para detectar cruces entre dos EMAs (media móvil exponencial).

    Args:
        df (pd.DataFrame): DataFrame base con columna de precio.
        fast (int): Periodo para la EMA rápida (ej: 12).
        slow (int): Periodo para la EMA lenta (ej: 26).
        price_col (str): Columna de precio sobre la que se calculan las EMAs.
        verbose (bool): Si True, imprime descripción de lo generado.

    Returns:
        tuple: 
            - df (DataFrame): con columnas nuevas 'ema_{fast}', 'ema_{slow}', y 'ema_cross_signal_{fast}_{slow}'
            - col_signal (str): nombre de la columna de señal de cruce

    Indicador:
        Detecta cambios de tendencia usando dos EMAs.
        Cuando la rápida cruza hacia arriba la lenta → señal de entrada (1).
        Cuando cruza hacia abajo → señal de salida (-1).
        Si no hay cruce → 0.
    """
    col_fast = f'ema_{fast}'
    col_slow = f'ema_{slow}'
    col_signal = f'ema_cross_signal_{fast}_{slow}'

    # Añade las EMAs si no existen ya
    if col_fast not in df.columns:
        df, _ = add_ema(df, period=fast, price_col=price_col, verbose=False)
    if col_slow not in df.columns:
        df, _ = add_ema(df, period=slow, price_col=price_col, verbose=False)

    # Señal de cruce
    cond_up = (df[col_fast] > df[col_slow]) & (df[col_fast].shift(1) <= df[col_slow].shift(1))
    cond_down = (df[col_fast] < df[col_slow]) & (df[col_fast].shift(1) >= df[col_slow].shift(1))

    df[col_signal] = 0
    df.loc[cond_up, col_signal] = 1   # cruce hacia arriba
    df.loc[cond_down, col_signal] = -1  # cruce hacia abajo

    if verbose:
        print(f"✅ Añadido {col_signal}: señales de cruce entre EMA{fast} y EMA{slow}. "
              f"1 = cruce alcista, -1 = cruce bajista, 0 = sin cruce.")

    return df, col_signal

def add_trade_outcome(df, horizon=24, take_profit=3, stop_loss=3):
    """
    Añade columna 'trade_outcome' con el resultado esperado del trade en las próximas 'horizon' velas.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'close', 'high', 'low'.
        horizon (int): Número de velas a mirar hacia el futuro.
        take_profit (float): Porcentaje de ganancia objetivo (ej 3 = 3%).
        stop_loss (float): Porcentaje de pérdida máxima permitida (ej 3 = 3%).

    Returns:
        tuple: (DataFrame modificado, nombre columna outcome)
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    length = len(df)

    outcomes = []

    for i in range(length):
        window_highs = highs[i+1:i+1+horizon]
        window_lows = lows[i+1:i+1+horizon]

        if len(window_highs) == 0:
            outcomes.append('ninguno')
            continue

        entry_price = closes[i]

        tp_hit_idx = None
        sl_hit_idx = None

        for idx, (h, l) in enumerate(zip(window_highs, window_lows)):
            gain = (h - entry_price) / entry_price * 100
            loss = (entry_price - l) / entry_price * 100

            if tp_hit_idx is None and gain >= take_profit:
                tp_hit_idx = idx
            if sl_hit_idx is None and loss >= stop_loss:
                sl_hit_idx = idx

            if tp_hit_idx is not None and sl_hit_idx is not None:
                break

        if tp_hit_idx is not None and sl_hit_idx is not None:
            if tp_hit_idx < sl_hit_idx:
                outcomes.append('take_profit')
            else:
                outcomes.append('stop_loss')
        elif tp_hit_idx is not None:
            outcomes.append('take_profit')
        elif sl_hit_idx is not None:
            outcomes.append('stop_loss')
        else:
            outcomes.append('ninguno')

    col_outcome = f'result_trade_outcome_{horizon}N_{take_profit}TP_{stop_loss}SL'
    df[col_outcome] = outcomes

    col_gain_bool = f'result_gain_{horizon}N_{take_profit}TP_{stop_loss}SL_bool'

    df[col_gain_bool] = df[col_outcome].apply(lambda x: x == 'take_profit')

    print(f"✅ Añadida columna {col_outcome}: resultado del trade en las próximas {horizon} velas.")
    print(f"✅ Añadida columna {col_gain_bool}: indica si se alcanzó el take profit.")
    print(f"   'take_profit' si el precio subió al menos {take_profit}% antes que la caída de {stop_loss}%.")
    print(f"   'stop_loss' si la caída ocurrió antes que la subida objetivo.")
    print(f"   'ninguno' si ninguna condición se cumplió en el horizonte.")

    return df, col_outcome, col_gain_bool

# Funcion para sacar la relacion entre las varibales y la salida binaria
from sklearn.feature_selection import mutual_info_classif

def calcular_importancia_features(df, verbose=True):
    """
    Calcula la importancia de cada feature numérica respecto a la variable binaria de salida
    usando información mutua (mutual_info_classif). Excluye columnas futuras (que empiecen por 'result_').

    Args:
        df (pd.DataFrame): DataFrame con variables técnicas y columna de salida.
        verbose (bool): Si True, imprime el ranking ordenado.

    Returns:
        List[Tuple[str, float]]: Lista ordenada de (feature, score).
    """
    # Detectar columna binaria de salida
    target_cols = [col for col in df.columns if col.startswith('result_gain_') and col.endswith('_bool')]
    if not target_cols:
        raise ValueError("❌ No se encontró ninguna columna de salida binaria con formato 'result_gain_..._bool'.")
    
    target_col = target_cols[0]  # cogemos la primera que cumpla

    # Filtrar columnas numéricas válidas (excluyendo las futuras)
    feature_cols = [
        col for col in df.select_dtypes(include='number').columns
        if not col.startswith('result_') and col != target_col
    ]

    # Extraer X e y
    X = df[feature_cols].fillna(0)
    y = df[target_col].astype(int)

    # Calcular mutual info
    importancias = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    # Ordenar y devolver
    ranking = sorted(zip(feature_cols, importancias), key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"✅ Importancia de features respecto a la salida binaria '{target_col}':\n")
        for nombre, score in ranking:
            print(f"   {nombre:<30} ➜ {score:.4f}")

    return ranking


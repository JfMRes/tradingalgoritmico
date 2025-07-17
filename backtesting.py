def backtesting(df_pred, capital_inicial=100, take_profit=3, stop_loss=1):

    df_completo = df_pred.copy()

    # Ordenamos por la columna date para asegurar secuencia temporal
    df_completo.sort_values('date', inplace=True)
    df_completo.reset_index(drop=True, inplace=True)  # Reset índice para evitar confusiones

    # Inicializamos columnas para control de posición y resultados
    df_completo['open_position'] = False
    df_completo['gains'] = 0.0
    df_completo['entry_price'] = None
    df_completo['exit_price'] = None
    df_completo['exit_reason'] = None

    open_position = False
    entry_price, value_take_profit, value_stop_loss = 0.0, 0.0, 0.0

    # Iteramos por índice posicional para controlar fila y poder acceder a la siguiente
    for indice in range(len(df_completo)):

        # Si no hay posición abierta y modelo indica oportunidad
        if not open_position and df_completo.loc[indice, 'model_pred']:
            # Abrimos posición sólo si no estamos en la última fila
            if indice + 1 < len(df_completo):
                entry_price = df_completo.loc[indice + 1, "open"]
                value_take_profit = entry_price * (1 + take_profit / 100)
                value_stop_loss = entry_price * (1 - stop_loss / 100)

                df_completo.loc[indice, 'open_position'] = True
                df_completo.loc[indice, 'entry_price'] = entry_price
                open_position = True

        elif open_position:
            # Evaluamos si stop loss o take profit han sido alcanzados
            low_price = df_completo.loc[indice, 'low']
            high_price = df_completo.loc[indice, 'high']

            if low_price <= value_stop_loss:
                # Stop loss alcanzado: calculamos pérdida
                df_completo.loc[indice, 'gains'] = (low_price - entry_price) / entry_price
                df_completo.loc[indice, 'exit_price'] = low_price
                df_completo.loc[indice, 'exit_reason'] = 'SL'
                df_completo.loc[indice, 'open_position'] = False

                entry_price, value_take_profit, value_stop_loss = 0.0, 0.0, 0.0
                open_position = False

            elif high_price >= value_take_profit:
                # Take profit alcanzado: calculamos ganancia
                df_completo.loc[indice, 'gains'] = (high_price - entry_price) / entry_price
                df_completo.loc[indice, 'exit_price'] = high_price
                df_completo.loc[indice, 'exit_reason'] = 'TP'
                df_completo.loc[indice, 'open_position'] = False

                entry_price, value_take_profit, value_stop_loss = 0.0, 0.0, 0.0
                open_position = False

            else:
                # No se cierra posición, se mantiene abierta
                df_completo.loc[indice, 'open_position'] = True

    # Si llegamos al final con posición abierta, la cerramos con el precio de cierre final
    if open_position:
        ultimo_indice = len(df_completo) - 1
        close_price = df_completo.loc[ultimo_indice, 'close']
        df_completo.loc[ultimo_indice, 'gains'] = (close_price - entry_price) / entry_price * 100
        df_completo.loc[ultimo_indice, 'exit_price'] = close_price
        df_completo.loc[ultimo_indice, 'exit_reason'] = 'End'
        df_completo.loc[ultimo_indice, 'open_position'] = False

    

    # Calculo de capital para cada caso. 
    df_completo['disponible'] = None

    for indice in range(len(df_completo)):
        if indice == 0:
            df_completo.loc[indice, 'disponible'] = capital_inicial
        else:
            df_completo.loc[indice, 'disponible'] = (
                1 + df_completo.loc[indice, 'gains']
            ) * df_completo.loc[indice - 1, 'disponible']


    return df_completo

def back_testing_resume(df):
    capital_inicial = df['disponible'].iloc[0]
    capital_final = df['disponible'].iloc[-1]
    diferencia = capital_final - capital_inicial
    diferencia_pct = (diferencia / capital_inicial) * 100

    print("----- Resumen Backtesting -----")
    print(f"Capital inicial: {capital_inicial:.2f}")
    print(f"Capital final: {capital_final:.2f}")
    print(f"Diferencia: {diferencia:.2f}")
    print(f"Diferencia porcentual: {diferencia_pct:.2f}%")

    print("----- Fechas -----")
    periodo_dias = (df['date'].max() - df['date'].min()).days or 1
    print(f"Periodo: desde {df['date'].min().date()} hasta {df['date'].max().date()} ({periodo_dias} días, {periodo_dias // 30} meses, {periodo_dias // 365} años)")
    media_ganancia_diaria = df['gains'].sum() / periodo_dias * 100
    media_ganancia_anual = media_ganancia_diaria * 365
    print(f"Media de ganancia diaria: {media_ganancia_diaria:.6f}%")
    print(f"Media de ganancia anual: {media_ganancia_anual:.6f}%")

    # Conteo de cada tipo de exit_reason y total
    counts = df['exit_reason'].value_counts(dropna=True)
    total_operaciones = counts.sum()
    print("Operaciones cerradas por tipo:")
    for tipo, cantidad in counts.items():
        print(f"  {tipo}: {cantidad}")
    print(f"Total operaciones cerradas: {total_operaciones}")

    print("--------------------------------")

from visualizacion import * 

def back_testing_graph(df):
    # Solo grafica la evolución del capital disponible (ganancias acumuladas)
    graficar(df, columnas=['disponible', 'open'], juntas=False, last_n=len(df))
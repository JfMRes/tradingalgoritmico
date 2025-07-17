def backtesting(df, df_pred, capital_inicial = 100):

    df = df.copy()
    df.reset_index(inplace=True, drop=True)

    df_pred = df_pred.copy()
    df_pred.reset_index(inplace=True, drop=True)

    df_completo = df.merge(df_pred, on='date', how='left')

    # Ordenamos por la columna date
    df_completo.sort_values('date', inplace=True)

    # Primero metemos en el df una columna de si esta la operacion abierta o no, en inicio false
    df_completo['open_position'] = False

    open_position = False
    value_take_profite, value_stop_loss = 0.0, 0.0
    # Luego iteramos para cada posicion

    for idx, row in df_completo.iterrows():

        if not open_position:
            if df_completo.loc[idx,'model_pred']:
                # Abrimos posicion
                

        
    
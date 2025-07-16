import pandas as pd

def lecturaYescritura():
    # Leer datos
    df = pd.read_csv('btcusd_1-min_data.csv')

    # Convertir timestamp a datetime
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True)
    df.set_index('date', inplace=True)

    # Renombrar columna Open a price
    df.rename(columns={'Open': 'price'}, inplace=True)

    # Crear OHLC a 10 minutos
    df_10m = df.resample('10T').agg({
        'price': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })

    # Renombrar columnas
    df_10m.rename(columns={
        'price': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Eliminar filas con datos faltantes (por huecos)
    df_10m.dropna(inplace=True)

    return df_10m.reset_index()  # Devolver con fecha como columna

if __name__ == "__main__":
    lecturaYescritura().to_csv('data/BTCUSDT.csv', index=False)

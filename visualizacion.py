import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

def graficar(df, columnas, last_n=1000, juntas=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import pandas as pd

    plt.close('all')
    plt.style.use('default')

    # Convertir 'date' a datetime, ignorar errores para no romper
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Eliminar filas con fecha NaT
    df = df.dropna(subset=['date'])

    # Ordenar por fecha y resetear índice
    df = df.sort_values('date').reset_index(drop=True)

    # Tomar últimas last_n filas
    data = df[['date'] + columnas].tail(last_n).copy()
    data.set_index('date', inplace=True)

    max_xticks = 20
    max_yticks = 20

    if juntas:
        scaled = data.copy()
        for col in columnas:
            min_val = scaled[col].min()
            max_val = scaled[col].max()
            if max_val != min_val:
                scaled[col] = (scaled[col] - min_val) / (max_val - min_val)
            else:
                scaled[col] = 0.5  # Si no varía, poner constante para que se vea

        fig, ax = plt.subplots(figsize=(12, 6))
        for col in columnas:
            ax.plot(scaled.index, scaled[col], label=col)

        ax.set_title(f'Últimas {last_n} filas de {", ".join(columnas)} (escaladas)')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor normalizado [0-1]')
        ax.legend()
        ax.grid(True)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_xticks))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_yticks))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    else:
        n = len(columnas)
        fig, axs = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
        if n == 1:
            axs = [axs]

        for i, col in enumerate(columnas):
            axs[i].plot(data.index, data[col], label=col)
            axs[i].set_title(f'{col} - Últimas {last_n} filas')
            axs[i].set_ylabel('Valor')
            axs[i].grid(True)
            axs[i].legend()
            axs[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_yticks))

        axs[-1].set_xlabel('Fecha')
        axs[-1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_xticks))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

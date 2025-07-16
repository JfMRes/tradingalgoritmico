import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def graficar(df, columnas, last_n=1000, juntas=True):
    plt.close('all')
    plt.style.use('default')

    data = df[['date'] + columnas].tail(last_n).copy()
    data.set_index('date', inplace=True)

    max_xticks = 20
    max_yticks = 20

    if juntas:
        scaled = data.copy()
        for col in columnas:
            min_val = scaled[col].min()
            max_val = scaled[col].max()
            scaled[col] = (scaled[col] - min_val) / (max_val - min_val)

        fig, ax = plt.subplots(figsize=(12, 6))
        for col in columnas:
            ax.plot(scaled.index, scaled[col], label=col)

        ax.set_title(f'Últimas {last_n} filas de {", ".join(columnas)} (escaladas)')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor normalizado [0-1]')

        ax.legend()
        ax.grid(True)

        # Limitar ticks en X
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_xticks))
        # Limitar ticks en Y
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

            # Limitar ticks en Y de cada subplot
            axs[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_yticks))

        axs[-1].set_xlabel('Fecha')

        # Limitar ticks en X solo en la última gráfica para que no se repita
        axs[-1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_xticks))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

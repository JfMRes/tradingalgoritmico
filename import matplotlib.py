import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dates = pd.date_range(start='2023-01-01', periods=50)
df = pd.DataFrame({
    'date': dates,
    'val1': np.random.randn(50).cumsum(),
    'val2': np.random.randn(50).cumsum()
})
df.set_index('date', inplace=True)

plt.style.use('default')  # Para usar estilo base limpio

fig, ax = plt.subplots()
ax.plot(df.index, df['val1'], label='val1')
ax.plot(df.index, df['val2'], label='val2')
ax.set_title('Test simple')
ax.set_xlabel('Fecha')
ax.set_ylabel('Valor')
ax.legend()
ax.grid(True)
plt.show()

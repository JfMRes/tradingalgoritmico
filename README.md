# tradingalgoritmico
## Funcionamientos

### Entrenamiento

1. Toma de datos.
1. Enriquecemos datos.
1. Partimos datos tomando datos reales del ultimo % del año. Esos datos eliminados del final los guardamos.
1. Obtenemos el resultado de para cada dato de entranamiento que hubiera pasado si hubiesemos invertido.
    1. Si en las proximas n velas, sube un x% antes de bajar un y% - Hemos ganado - True.
    1. Si antes baja un y% o ni llega a bajar y& ni a subir x% - Hemos perdido - False.
1. Por curiosidad, vemos la relacion de los parametros con la salida binaria.
1. Balanceamos datos, ya que False aparece muchas mas veces que true.
1. Creamos el modelo (RandomForest por ejemplo)
1. Realizamos predicciones 
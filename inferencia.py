def predict_from_model(df, model, feature_cols, threshold=0.5, return_probs=True):
    """
    Aplica un modelo entrenado a un DataFrame con las columnas necesarias.

    Args:
        df (pd.DataFrame): Datos ya preparados con features.
        model: Modelo ya entrenado (ej. RandomForest).
        feature_cols (list): Lista de nombres de columnas a usar como input.
        threshold (float): Umbral de probabilidad para decidir True/False.
        return_probs (bool): Si True, añade columna con probabilidades.

    Returns:
        pd.Series (predicciones True/False)
        Si return_probs=True, también añade columna 'pred_proba'
    """
    df = df.copy()
    X = df[feature_cols].fillna(0)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        preds = probs >= threshold
        if return_probs:
            df['pred_proba'] = probs
    else:
        preds = model.predict(X)
        if return_probs:
            df['pred_proba'] = preds  # En este caso no hay proba real

    df['model_pred'] = preds.astype(bool)
    
    if return_probs:
        return df
    else:
        return df['model_pred']
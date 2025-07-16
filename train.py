from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def clean_train(df, test_size=0.2, random_state=42):
    """
    Limpia y prepara los datos para entrenamiento y test.

    Args:
        df (pd.DataFrame): DataFrame original con features y target.
        test_size (float): Proporción para test.
        random_state (int): Semilla reproducible.

    Returns:
        X_train, X_test, y_train, y_test: Arrays de entrenamiento y prueba.
        feature_cols (list): Lista de columnas usadas como features.
        target_col (str): Nombre de la columna target binaria.
    """
    # Detectar columna target (igual que antes)
    target_cols = [col for col in df.columns if col.startswith('result_gain_') and col.endswith('_bool')]
    if not target_cols:
        raise ValueError("No se encontró columna target binaria con formato esperado.")
    target_col = target_cols[0]

    # Filtrar features numéricas válidas
    feature_cols = [
        col for col in df.select_dtypes(include='number').columns
        if not col.startswith('result_') and col != target_col
    ]

    X = df[feature_cols].fillna(0)
    y = df[target_col].astype(int)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols, target_col


def balanced_methods(df, method='undersample', random_state=42):
    import numpy as np
    np.random.seed(random_state)

    target_cols = [col for col in df.columns if col.startswith('result_gain_') and col.endswith('_bool')]
    if not target_cols:
        raise ValueError("No se encontró columna target binaria con formato esperado.")
    target_col = target_cols[0]

    if method == 'undersample':
        df_majority = df[df[target_col] == 0]
        df_minority = df[df[target_col] == 1]

        n_minority = len(df_minority)
        df_majority_downsampled = df_majority.sample(n=n_minority, random_state=random_state)

        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

        return df_balanced, target_col

    else:
        raise NotImplementedError(f"Método '{method}' no implementado todavía.")


def execute_random_forest(df, n_estimators=100, random_state=42, balanced = False):
    """
    Entrena y evalúa un RandomForestClassifier sobre los datos.

    Args:
        df (pd.DataFrame): DataFrame con features y target.
        n_estimators (int): Número de árboles en el bosque.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        None
    """
    X_train, X_test, y_train, y_test, feature_cols, target_col = clean_train(df)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"✅ RandomForest entrenado con {n_estimators} árboles.")
    print(f"Features usadas: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Tamaño train: {len(X_train)}, test: {len(X_test)}")
    print("Resultados en test:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
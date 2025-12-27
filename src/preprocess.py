import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    input_path="data/raw/dados_credito_sinteticos.csv",
    output_path="data/processed/dados_credito_processados.csv",
    return_df=False
):
    df = pd.read_csv(input_path)

    # Separar target
    y = df["aprovado_credito"]

    # Separar features numéricas (EXCLUINDO client_id)
    X = df.drop(["aprovado_credito", "client_id"], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dataset processado
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["aprovado_credito"] = y.values
    df_processed["client_id"] = df["client_id"].values

    df_processed.to_csv(output_path, index=False)

    # Split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X_scaled,
        y,
        df[["client_id"]].reset_index(drop=True),
        test_size=0.2,
        random_state=42
    )

    if return_df:
        return X_train, X_test, y_train, y_test, df_test

    return X_train, X_test, y_train, y_test

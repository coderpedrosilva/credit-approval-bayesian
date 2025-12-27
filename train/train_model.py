import os
from src.preprocess import preprocess_data
from src.bayesian_logistic import train_bayesian_logistic

TRACE_PATH = "models/bayesian_credit_trace.nc"

def train_and_persist():
    os.makedirs("models", exist_ok=True)

    print("🔬 Treinando modelo bayesiano...")

    X_train, X_test, y_train, y_test = preprocess_data()
    model, trace = train_bayesian_logistic(X_train, y_train)

    trace.to_netcdf(TRACE_PATH)
    print("💾 Trace bayesiano salvo com sucesso.")

    return trace

import os
import joblib
from src.preprocess import preprocess_data
from src.bayesian_logistic import train_bayesian_logistic

MODEL_PATH = "models/bayesian_credit_model.pkl"
TRACE_PATH = "models/bayesian_credit_trace.nc"

os.makedirs("models", exist_ok=True)

print("Treinando modelo bayesiano...")

X_train, X_test, y_train, y_test = preprocess_data()

model, trace = train_bayesian_logistic(X_train, y_train)

joblib.dump(model, MODEL_PATH)
trace.to_netcdf(TRACE_PATH)

print("Modelo treinado e salvo com sucesso.")

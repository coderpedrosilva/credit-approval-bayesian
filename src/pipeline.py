import os

from src.generate_data import generate_dataset
from src.preprocess import preprocess_data
from src.naive_bayes import train_naive_bayes
from src.bayesian_logistic import train_bayesian_logistic
from src.evaluate_models import (
    evaluate_naive_bayes,
    evaluate_bayesian_logistic
)

from src.utils.save_results import save_metrics
from src.utils.save_trace import save_bayesian_trace
from src.interpretation.coefficients import summarize_and_save_coefficients


def ensure_directories():
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "results",
        "models"   
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def run_pipeline():
    print("🔹 Iniciando pipeline de crédito bayesiano...")

    # 1. Garantir estrutura
    ensure_directories()

    # 2. Dados
    generate_dataset()

    # 3. Pré-processamento
    X_train, X_test, y_train, y_test = preprocess_data()

    # 4. Baseline
    nb_model = train_naive_bayes(X_train, y_train)
    nb_metrics = evaluate_naive_bayes(nb_model, X_test, y_test)

    # 5. Modelo bayesiano
    bayes_model, trace = train_bayesian_logistic(X_train, y_train)
    bayes_metrics = evaluate_bayesian_logistic(trace, X_test, y_test)

    # 6. Persistência oficial
    save_bayesian_trace(trace)          # → models/bayesian_credit_trace.nc
    save_metrics([nb_metrics, bayes_metrics])
    coef_summary = summarize_and_save_coefficients(trace)

    # 7. Terminal
    print("\n📊 Resultados:")
    print(nb_metrics)
    print(bayes_metrics)

    print("\n📈 Interpretação dos coeficientes bayesianos:")
    print(coef_summary)

    print("\n💾 Modelo bayesiano persistido em models/")
    print("\n✅ Pipeline finalizado com sucesso!")

import os

from src.generate_data import generate_dataset
from src.preprocess import preprocess_data
from src.naive_bayes import train_naive_bayes
from src.bayesian_logistic import train_bayesian_logistic
from src.evaluate_models import (
    evaluate_naive_bayes,
    evaluate_bayesian_logistic
)


def ensure_directories():
    """
    Cria a estrutura de diretórios necessária caso não exista.
    Ideal quando a pasta data/ está no .gitignore.
    """
    directories = [
        "data",
        "data/raw",
        "data/processed"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def run_pipeline():
    print("🔹 Iniciando pipeline de crédito bayesiano...")

    # 1. Garantir estrutura de pastas
    ensure_directories()

    # 2. Geração de dados
    generate_dataset()

    # 3. Pré-processamento
    X_train, X_test, y_train, y_test = preprocess_data()

    # 4. Modelo baseline
    nb_model = train_naive_bayes(X_train, y_train)
    nb_metrics = evaluate_naive_bayes(nb_model, X_test, y_test)

    # 5. Modelo bayesiano
    bayes_model, trace = train_bayesian_logistic(X_train, y_train)
    bayes_metrics = evaluate_bayesian_logistic(trace, X_test, y_test)

    # 6. Resultados
    print("\n📊 Resultados:")
    print(nb_metrics)
    print(bayes_metrics)

    print("\n✅ Pipeline finalizado com sucesso!")

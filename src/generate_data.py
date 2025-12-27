import numpy as np
import pandas as pd

def generate_dataset(n_samples=1500, output_path="data/raw/dados_credito_sinteticos.csv"):
    np.random.seed(42)

    idade = np.random.randint(18, 75, n_samples)

    renda_mensal = np.clip(
        np.random.normal(6000, 3000, n_samples),
        1000,
        25000
    )

    score_credito = np.clip(
        np.random.normal(650, 80, n_samples),
        300,
        850
    )

    valor_solicitado = np.clip(
        np.random.normal(15000, 8000, n_samples),
        1000,
        60000
    )

    historico_inadimplencia = np.random.choice(
        [0, 1],
        size=n_samples,
        p=[0.75, 0.25]
    )

    taxa_endividamento = np.random.uniform(0.1, 1.2, n_samples)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    prob_aprovacao = (
        -2.0
        + 0.004 * (score_credito - 600)
        + 0.00004 * renda_mensal
        - 1.5 * historico_inadimplencia
        - 2.0 * taxa_endividamento
        - 0.00003 * valor_solicitado
    )

    prob_aprovacao = sigmoid(prob_aprovacao)

    aprovado_credito = np.random.binomial(1, prob_aprovacao)

    df = pd.DataFrame({
        "idade": idade,
        "renda_mensal": renda_mensal.round(2),
        "score_credito": score_credito.round(0),
        "valor_solicitado": valor_solicitado.round(2),
        "taxa_endividamento": taxa_endividamento.round(2),
        "historico_inadimplencia": historico_inadimplencia,
        "aprovado_credito": aprovado_credito
    })

    df.to_csv(output_path, index=False)
    print("✅ Dataset sintético gerado com sucesso!")

if __name__ == "__main__":
    generate_dataset()

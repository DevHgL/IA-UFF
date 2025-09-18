# ======================================
# Imports
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ======================================
# Carregamento do dataset
# ======================================
df = pd.read_csv("housing.csv")

# ======================================
# Pré-processamento
# ======================================

# 1. Remover valores nulos
df = df.dropna()

# 2. Feature Engineering simples
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# 3. Log-transform em variáveis com outliers pesados
for col in ["total_rooms", "total_bedrooms", "population", "households"]:
    df[col] = np.log1p(df[col])

# 4. One-hot encoding da variável categórica
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# 5. Separar features (X) e target (y)
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# 6. Escalonamento robusto (menos sensível a outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 7. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ======================================
# Modelos (apenas regressões lineares)
# ======================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)
}

results = {}
predictions = {}

# ======================================
# Treino e avaliação
# ======================================
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape  # aproximação de "acurácia" em %

    # Guardar resultados
    results[name] = {
        "RMSE": rmse,
        "R²": r2,
        "MAE": mae,
        "MAPE (%)": mape,
        "Accuracy (%)": accuracy
    }
    predictions[name] = y_pred

# ======================================
# Resultados numéricos
# ======================================
print("\n=== Resultados ===")
for name, metrics in results.items():
    print(f"\n{name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# ======================================
# Visualizações (salvas em PNG)
# ======================================

# Gráfico 1: Previsões vs Valores Reais (Linear Regression)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=predictions["Linear Regression"], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.title("Valores Reais vs Previsões (Linear Regression)")
plt.savefig("grafico_linear_regression.png", dpi=300, bbox_inches="tight")
plt.close()

# Gráfico 2: Distribuição dos Resíduos (Ridge Regression)
residuos = y_test - predictions["Ridge Regression"]
plt.figure(figsize=(8,6))
sns.histplot(residuos, bins=50, kde=True)
plt.xlabel("Resíduos")
plt.title("Distribuição dos Resíduos (Ridge Regression)")
plt.savefig("grafico_residuos_ridge.png", dpi=300, bbox_inches="tight")
plt.close()

# Gráfico 3: Comparação de métricas
metrics_df = pd.DataFrame(results).T
metrics_df.plot(kind="bar", figsize=(12,6))
plt.title("Comparação de Desempenho entre Modelos")
plt.ylabel("Valor da Métrica")
plt.xticks(rotation=0)
plt.savefig("grafico_comparacao_metricas.png", dpi=300, bbox_inches="tight")
plt.close()

# ======================================
# Salvar resultados em CSV
# ======================================
results_df = pd.DataFrame(results).T
results_df.to_csv("resultados_modelos.csv", index=True)

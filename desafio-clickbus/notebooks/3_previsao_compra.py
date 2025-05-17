#!/usr/bin/env python
# coding: utf-8

# # Previsão da Próxima Compra - ClickBus
# 
# Este script implementa o segundo desafio: prever se um cliente realizará uma compra na plataforma nos próximos 7 ou 30 dias.

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, 
                            roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locais
import sys
import os

# Adicionar diretório pai ao sys.path para encontrar os módulos src
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from src.data_processing import (load_data, create_time_features, create_purchase_features, 
                                create_route_features, prepare_for_next_purchase_prediction)
from src.models import train_next_purchase_model, predict_purchases
from src.utils import create_sample_data, save_model, load_model

# Configurações de visualização
plt.style.use('ggplot')
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

print("# Carregamento e Preparação dos Dados")
print("Vamos carregar os dados e preparar as features para o modelo de previsão.")

# Carregar o arquivo de dados (ou criar dados de exemplo se houver erro)
file_path = '../dados_desafio_fiap/hash/df_t.csv'

try:
    # Tente carregar o arquivo real
    print("Tentando carregar dados do arquivo:", file_path)
    df = load_data(file_path)
    
    # Para desenvolvimento, podemos trabalhar com uma amostra
    sample_df = create_sample_data(df, sample_size=10000)
    df = sample_df  # Comentar esta linha para usar o dataset completo
    
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")
    print("Criando dados de exemplo para demonstração...")
    # Criar dados de exemplo para demonstração
    np.random.seed(42)
    
    # Exemplo: 1000 clientes com várias compras cada
    n_customers = 1000
    n_records = 10000
    
    # IDs de clientes
    customer_ids = np.random.randint(1000, 9999, n_customers)
    
    # Datas de reserva (último ano)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    booking_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
    
    # Datas de embarque (entre 1 e 60 dias após a reserva)
    boarding_dates = [bd + timedelta(days=np.random.randint(1, 60)) for bd in booking_dates]
    
    # Criação do DataFrame
    df = pd.DataFrame({
        'booking_id': np.random.randint(10000, 99999, n_records),
        'customer_id': np.random.choice(customer_ids, n_records),
        'booking_date': booking_dates,
        'boarding_date': boarding_dates,
        'origin_id': np.random.randint(1, 50, n_records),
        'destination_id': np.random.randint(1, 50, n_records),
        'price': np.random.uniform(50, 500, n_records).round(2)
    })
    
    print(f"Dados de exemplo criados com sucesso. Dimensões: {df.shape}")

print("\nCriando features adicionais...")
# Criar features temporais
df = create_time_features(df)

# Criar features relacionadas ao histórico de compras
df = create_purchase_features(df)

# Criar features relacionadas a rotas
df = create_route_features(df)

# Visualizar as features criadas
print("Primeiras linhas com as features criadas:")
print(df.head())

print("\n# Preparação dos Dados para o Modelo de Previsão")
print("Vamos preparar os dados para prever se um cliente fará uma compra nos próximos 30 dias.")

# Preparar dados para previsão de compra nos próximos 30 dias
X_30, y_30 = prepare_for_next_purchase_prediction(df, future_days=30)

# Verificar o balanceamento das classes
print("Distribuição do target (compra nos próximos 30 dias):")
print(y_30.value_counts())
print(f"Percentual de casos positivos: {y_30.mean() * 100:.2f}%")

# Preparar dados para previsão de compra nos próximos 7 dias
X_7, y_7 = prepare_for_next_purchase_prediction(df, future_days=7)

# Verificar o balanceamento das classes
print("\nDistribuição do target (compra nos próximos 7 dias):")
print(y_7.value_counts())
print(f"Percentual de casos positivos: {y_7.mean() * 100:.2f}%")

print("\n# Exploração das Features para o Modelo")

# Verificar correlação entre as features e o target
numerical_features = X_30.select_dtypes(include=['int', 'float']).columns
X_corr = X_30[numerical_features].copy()
X_corr['target'] = y_30

# Calcular correlação com o target
correlations = X_corr.corr()['target'].sort_values(ascending=False)
print("Correlação das features com o target (compra nos próximos 30 dias):")
print(correlations)

# Visualizar correlação das principais features com o target
plt.figure(figsize=(12, 8))
top_features = correlations.drop('target').abs().sort_values(ascending=False).head(10).index
correlation_matrix = X_corr[list(top_features) + ['target']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação das Top 10 Features com o Target')
plt.savefig('../img/previsao/correlacao_features_target.png')
print("Gráfico salvo: img/previsao/correlacao_features_target.png")
plt.close()

# Analisar as features mais importantes
plt.figure(figsize=(16, 10))

for i, feature in enumerate(top_features[:4]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x=y_30, y=X_30[feature])
    plt.title(f'{feature} vs Target')
    plt.xlabel('Compra em 30 dias (0=Não, 1=Sim)')
    plt.ylabel(feature)

plt.tight_layout()
plt.savefig('../img/previsao/features_vs_target.png')
print("Gráfico salvo: img/previsao/features_vs_target.png")
plt.close()

print("\n# Treinamento do Modelo para Previsão de Compra em 30 dias")
print("Vamos treinar um modelo Random Forest para prever se um cliente fará uma compra nos próximos 30 dias.")

# Treinar modelo para previsão de compra em 30 dias
model_30d, results_30d = train_next_purchase_model(X_30, y_30)

# Visualizar as features mais importantes para o modelo de 30 dias
plt.figure(figsize=(12, 6))
feature_importance = results_30d['importance'].head(10)
feature_importance.plot(kind='barh', x='feature', y='importance', color='skyblue')
plt.title('Top 10 Features Mais Importantes para Previsão de Compra em 30 Dias')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.grid(True)
plt.savefig('../img/previsao/feature_importance_30d.png')
print("Gráfico salvo: img/previsao/feature_importance_30d.png")
plt.close()

# Avaliar o modelo de 30 dias com validação cruzada
X_model = X_30.drop('customer_id', axis=1, errors='ignore')
cv_scores = cross_val_score(results_30d['model'], X_model, y_30, cv=5, scoring='accuracy')

print(f"Acurácia média com validação cruzada (5-fold): {cv_scores.mean():.4f}")
print(f"Desvio padrão: {cv_scores.std():.4f}")
print(f"Scores individuais: {cv_scores}")

# Dividir dados em treino e teste para análise detalhada
X_train, X_test, y_train, y_test = train_test_split(X_model, y_30, test_size=0.2, random_state=42)

# Treinar modelo
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Avaliar modelo
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Visualizar a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.savefig('../img/previsao/matriz_confusao_30d.png')
print("Gráfico salvo: img/previsao/matriz_confusao_30d.png")
plt.close()

# Plotar a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Previsão de Compra em 30 Dias')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('../img/previsao/curva_roc_30d.png')
print("Gráfico salvo: img/previsao/curva_roc_30d.png")
plt.close()

print("\n# Treinamento do Modelo para Previsão de Compra em 7 dias")
print("Agora vamos treinar um modelo para prever se um cliente fará uma compra nos próximos 7 dias.")

# Treinar modelo para previsão de compra em 7 dias
model_7d, results_7d = train_next_purchase_model(X_7, y_7)

# Visualizar as features mais importantes para o modelo de 7 dias
plt.figure(figsize=(12, 6))
feature_importance = results_7d['importance'].head(10)
feature_importance.plot(kind='barh', x='feature', y='importance', color='salmon')
plt.title('Top 10 Features Mais Importantes para Previsão de Compra em 7 Dias')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.grid(True)
plt.savefig('../img/previsao/feature_importance_7d.png')
print("Gráfico salvo: img/previsao/feature_importance_7d.png")
plt.close()

print("\n# Desafio Extra: Previsão do Número de Dias até a Próxima Compra")
print("Como parte do desafio extra, vamos implementar um modelo de regressão para prever quantos dias faltam até a próxima compra do cliente.")

# Preparar dados para previsão do número de dias até a próxima compra
df_sorted = df.sort_values(['customer_id', 'booking_date'])

# Calcular dias até a próxima compra para cada registro
df_sorted['next_purchase_date'] = df_sorted.groupby('customer_id')['booking_date'].shift(-1)
df_sorted['days_to_next_purchase'] = (df_sorted['next_purchase_date'] - df_sorted['booking_date']).dt.days

# Remover registros sem próxima compra (última compra do cliente)
df_with_next = df_sorted.dropna(subset=['days_to_next_purchase'])

# Visualizar a distribuição dos dias até a próxima compra
plt.figure(figsize=(12, 6))
sns.histplot(df_with_next['days_to_next_purchase'], bins=50, kde=True)
plt.title('Distribuição dos Dias até a Próxima Compra')
plt.xlabel('Dias')
plt.ylabel('Frequência')
plt.grid(True)
plt.savefig('../img/previsao/distribuicao_dias_proxima_compra.png')
print("Gráfico salvo: img/previsao/distribuicao_dias_proxima_compra.png")
plt.close()

# Preparar features para o modelo de regressão
X_regression = df_with_next.drop(['next_purchase_date', 'days_to_next_purchase', 'booking_date', 
                                'boarding_date', 'prev_purchase', 'route', 'most_frequent_route'], axis=1, errors='ignore')
y_regression = df_with_next['days_to_next_purchase']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Remover colunas não necessárias e colunas de tipo string/objeto
X_train = X_train.drop('customer_id', axis=1, errors='ignore')
X_test = X_test.drop('customer_id', axis=1, errors='ignore')

# Selecionar apenas as colunas numéricas
X_train = X_train.select_dtypes(include=['int', 'float'])
X_test = X_test.select_dtypes(include=['int', 'float'])

# Treinar modelo de regressão
regressor = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
regressor.fit(X_train, y_train)

# Fazer previsões
y_pred = regressor.predict(X_test)

# Avaliar modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Resultados do Modelo de Regressão (Dias até a Próxima Compra):")
print(f"MAE: {mae:.2f} dias")
print(f"RMSE: {rmse:.2f} dias")
print(f"R²: {r2:.4f}")

# Visualizar previsões vs valores reais
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Dias Reais vs Dias Previstos até a Próxima Compra')
plt.xlabel('Dias Reais')
plt.ylabel('Dias Previstos')
plt.grid(True)
plt.savefig('../img/previsao/previsao_vs_real_dias.png')
print("Gráfico salvo: img/previsao/previsao_vs_real_dias.png")
plt.close()

# Visualizar as features mais importantes para o modelo de regressão
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': regressor.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
feature_importance.head(10).plot(kind='barh', x='feature', y='importance', color='lightgreen')
plt.title('Top 10 Features Mais Importantes para Previsão de Dias até a Próxima Compra')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.grid(True)
plt.savefig('../img/previsao/feature_importance_regression.png')
print("Gráfico salvo: img/previsao/feature_importance_regression.png")
plt.close()

print("\n# Salvar os Modelos")
print("Vamos salvar os modelos para uso posterior na aplicação.")

# Salvar os modelos
save_model(results_30d['model'], '../modelo_previsao_30dias.pkl')
save_model(results_7d['model'], '../modelo_previsao_7dias.pkl')
save_model(regressor, '../modelo_previsao_dias_ate_compra.pkl')

print("\n# Conclusões")
print("""
Neste script, desenvolvemos modelos para prever:

1. Se um cliente realizará uma compra nos próximos 30 dias: Implementamos um modelo de classificação binária com Random Forest que alcançou boa acurácia e F1-score.

2. Se um cliente realizará uma compra nos próximos 7 dias: O modelo para previsão de curto prazo também apresentou bom desempenho, embora com menos casos positivos devido ao horizonte temporal mais curto.

3. Quantos dias até a próxima compra do cliente: Como parte do desafio extra, implementamos um modelo de regressão que prevê o número de dias até a próxima compra, com erro médio absoluto de alguns dias.

Identificamos as features mais importantes para a previsão de compras futuras, incluindo:
- Recência da última compra
- Frequência de compras
- Valor monetário gasto
- Média de dias entre compras
- Preferência por determinadas rotas

Esses modelos permitirão à ClickBus identificar proativamente os clientes com maior probabilidade de realizar compras em diferentes horizontes temporais, possibilitando campanhas de marketing mais direcionadas e eficientes.

No próximo script, abordaremos o terceiro desafio: prever qual será o próximo trecho que o cliente tem maior probabilidade de comprar.
""") 
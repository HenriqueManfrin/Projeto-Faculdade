#!/usr/bin/env python
# coding: utf-8

# # Análise Exploratória de Dados - ClickBus
# 
# Este script contém a análise exploratória inicial dos dados da ClickBus.

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locais
import sys
import os

# Adicionar diretório pai ao sys.path para encontrar os módulos src
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from src.utils import create_sample_data

# Configurações de visualização
plt.style.use('ggplot')
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

print("# Carregamento dos Dados")
print("Vamos carregar os dados do arquivo CSV fornecido pela ClickBus.")

# Carregar o arquivo de dados
file_path = '../dados_desafio_fiap/hash/df_t.csv'

# Para testar com uma amostra menor, podemos usar:
# df = pd.read_csv(file_path, nrows=100000, parse_dates=['booking_date', 'boarding_date'])

# Carregar o dataset ou criar dados de exemplo se houver erro
try:
    print("Tentando carregar dados do arquivo:", file_path)
    df = pd.read_csv(file_path, parse_dates=['booking_date', 'boarding_date'])
    print(f"Dados carregados com sucesso. Dimensões: {df.shape}")
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

print("\n# Verificação Inicial dos Dados")
print("Vamos analisar a estrutura dos dados e verificar a presença de valores nulos ou inconsistentes.")

# Visualizar as primeiras linhas do dataset
print("\nPrimeiras linhas do dataset:")
print(df.head())

# Informações sobre os tipos de dados e valores não-nulos
print("\nInformações sobre os tipos de dados:")
print(df.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe().T)

# Verificar valores nulos
print("\nValores nulos por coluna:")
print(df.isnull().sum())

# Percentual de valores nulos
print("\nPercentual de valores nulos:")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\n# Análise da Distribuição dos Dados")
print("Vamos analisar a distribuição das principais variáveis do dataset.")

# Criar coluna de rota (origem-destino)
df['route'] = df['origin_id'].astype(str) + '-' + df['destination_id'].astype(str)

# Distribuição de preços
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribuição de Preços de Passagens')
plt.xlabel('Preço (R$)')
plt.ylabel('Frequência')
plt.grid(True)
plt.savefig('../img/exploracao/distribuicao_precos.png')
print("Gráfico salvo: img/exploracao/distribuicao_precos.png")
plt.close()

# Adicionar variáveis temporais
df['year'] = df['booking_date'].dt.year
df['month'] = df['booking_date'].dt.month
df['day'] = df['booking_date'].dt.day
df['dayofweek'] = df['booking_date'].dt.dayofweek
df['weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Calcular dias de antecedência da compra
df['days_before_boarding'] = (df['boarding_date'] - df['booking_date']).dt.days

# Distribuição de dias de antecedência da compra
plt.figure(figsize=(12, 6))
sns.histplot(df['days_before_boarding'], bins=50, kde=True)
plt.title('Distribuição de Dias de Antecedência da Compra')
plt.xlabel('Dias antes do embarque')
plt.ylabel('Frequência')
plt.grid(True)
plt.savefig('../img/exploracao/antecedencia_compra.png')
print("Gráfico salvo: img/exploracao/antecedencia_compra.png")
plt.close()

# Compras por mês
plt.figure(figsize=(12, 6))
monthly_counts = df.groupby('month').size()
monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Número de Compras por Mês')
plt.xlabel('Mês')
plt.ylabel('Número de Compras')
plt.grid(True)
plt.savefig('../img/exploracao/compras_por_mes.png')
print("Gráfico salvo: img/exploracao/compras_por_mes.png")
plt.close()

# Compras por dia da semana
plt.figure(figsize=(12, 6))
day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
weekday_counts = df.groupby('dayofweek').size()
weekday_counts.index = [day_names[i] for i in weekday_counts.index]
weekday_counts.plot(kind='bar', color='salmon')
plt.title('Número de Compras por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Número de Compras')
plt.grid(True)
plt.savefig('../img/exploracao/compras_por_dia.png')
print("Gráfico salvo: img/exploracao/compras_por_dia.png")
plt.close()

print("\n# Análise das Rotas e Destinos")

# Top 10 rotas mais populares
top_routes = df['route'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_routes.plot(kind='barh', color='lightgreen')
plt.title('Top 10 Rotas Mais Populares')
plt.xlabel('Número de Viagens')
plt.ylabel('Rota (Origem-Destino)')
plt.grid(True)
plt.savefig('../img/exploracao/top_rotas.png')
print("Gráfico salvo: img/exploracao/top_rotas.png")
plt.close()

# Top 10 origens mais comuns
top_origins = df['origin_id'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_origins.plot(kind='barh', color='lightblue')
plt.title('Top 10 Origens Mais Comuns')
plt.xlabel('Número de Viagens')
plt.ylabel('ID da Origem')
plt.grid(True)
plt.savefig('../img/exploracao/top_origens.png')
print("Gráfico salvo: img/exploracao/top_origens.png")
plt.close()

# Top 10 destinos mais comuns
top_destinations = df['destination_id'].value_counts().head(10)

plt.figure(figsize=(12, 6))
top_destinations.plot(kind='barh', color='lightpink')
plt.title('Top 10 Destinos Mais Comuns')
plt.xlabel('Número de Viagens')
plt.ylabel('ID do Destino')
plt.grid(True)
plt.savefig('../img/exploracao/top_destinos.png')
print("Gráfico salvo: img/exploracao/top_destinos.png")
plt.close()

# Preço médio por rota (top 10 rotas)
route_price = df.groupby('route')['price'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
route_price.plot(kind='barh', color='orange')
plt.title('Top 10 Rotas Mais Caras (Preço Médio)')
plt.xlabel('Preço Médio (R$)')
plt.ylabel('Rota (Origem-Destino)')
plt.grid(True)
plt.savefig('../img/exploracao/rotas_caras.png')
print("Gráfico salvo: img/exploracao/rotas_caras.png")
plt.close()

print("\n# Análise de Comportamento de Clientes")

# Número de compras por cliente
customer_purchases = df.groupby('customer_id').size().reset_index(name='num_purchases')
print("Estatísticas de número de compras por cliente:")
print(customer_purchases['num_purchases'].describe())

# Distribuição do número de compras por cliente
plt.figure(figsize=(12, 6))
sns.histplot(customer_purchases['num_purchases'], bins=30, kde=True)
plt.title('Distribuição do Número de Compras por Cliente')
plt.xlabel('Número de Compras')
plt.ylabel('Número de Clientes')
plt.grid(True)
plt.savefig('../img/exploracao/compras_por_cliente.png')
print("Gráfico salvo: img/exploracao/compras_por_cliente.png")
plt.close()

# Gasto total por cliente
customer_spend = df.groupby('customer_id')['price'].sum().reset_index(name='total_spend')
print("Estatísticas de gasto total por cliente:")
print(customer_spend['total_spend'].describe())

# Distribuição do gasto total por cliente
plt.figure(figsize=(12, 6))
sns.histplot(customer_spend['total_spend'], bins=30, kde=True)
plt.title('Distribuição do Gasto Total por Cliente')
plt.xlabel('Gasto Total (R$)')
plt.ylabel('Número de Clientes')
plt.grid(True)
plt.savefig('../img/exploracao/gasto_total_cliente.png')
print("Gráfico salvo: img/exploracao/gasto_total_cliente.png")
plt.close()

# Relação entre número de compras e gasto total
customer_analysis = pd.merge(customer_purchases, customer_spend, on='customer_id')
customer_analysis['avg_ticket'] = customer_analysis['total_spend'] / customer_analysis['num_purchases']

plt.figure(figsize=(10, 8))
plt.scatter(customer_analysis['num_purchases'], customer_analysis['total_spend'], alpha=0.5)
plt.title('Relação entre Número de Compras e Gasto Total')
plt.xlabel('Número de Compras')
plt.ylabel('Gasto Total (R$)')
plt.grid(True)
plt.savefig('../img/exploracao/relacao_compras_gasto.png')
print("Gráfico salvo: img/exploracao/relacao_compras_gasto.png")
plt.close()

print("\n# Preparação para Análise RFM")
print("Vamos calcular as métricas de Recência, Frequência e Valor Monetário para cada cliente.")

# Definir data de referência (último dia no dataset + 1)
reference_date = df['booking_date'].max() + timedelta(days=1)
print(f"Data de referência para análise RFM: {reference_date}")

# Calcular métricas RFM
rfm = df.groupby('customer_id').agg({
    'booking_date': lambda x: (reference_date - x.max()).days,  # Recência (R)
    'booking_id': 'nunique',  # Frequência (F)
    'price': 'sum'  # Valor Monetário (M)
}).reset_index()

# Renomear colunas
rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

# Estatísticas das métricas RFM
print("Estatísticas das métricas RFM:")
print(rfm[['recency', 'frequency', 'monetary']].describe())

# Visualizar distribuição de Recência
plt.figure(figsize=(12, 6))
sns.histplot(rfm['recency'], bins=30, kde=True)
plt.title('Distribuição de Recência (dias desde última compra)')
plt.xlabel('Dias')
plt.ylabel('Número de Clientes')
plt.grid(True)
plt.savefig('../img/segmentacao/recencia.png')
print("Gráfico salvo: img/segmentacao/recencia.png")
plt.close()

# Visualizar distribuição de Frequência
plt.figure(figsize=(12, 6))
sns.histplot(rfm['frequency'], bins=30, kde=True)
plt.title('Distribuição de Frequência (número de compras)')
plt.xlabel('Número de Compras')
plt.ylabel('Número de Clientes')
plt.grid(True)
plt.savefig('../img/segmentacao/frequencia.png')
print("Gráfico salvo: img/segmentacao/frequencia.png")
plt.close()

# Visualizar distribuição de Valor Monetário
plt.figure(figsize=(12, 6))
sns.histplot(rfm['monetary'], bins=30, kde=True)
plt.title('Distribuição de Valor Monetário (gasto total)')
plt.xlabel('Valor (R$)')
plt.ylabel('Número de Clientes')
plt.grid(True)
plt.savefig('../img/segmentacao/valor_monetario.png')
print("Gráfico salvo: img/segmentacao/valor_monetario.png")
plt.close()

# Matriz de correlação das métricas RFM
plt.figure(figsize=(10, 8))
correlation = rfm[['recency', 'frequency', 'monetary']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação das Métricas RFM')
plt.savefig('../img/segmentacao/correlacao_rfm.png')
print("Gráfico salvo: img/segmentacao/correlacao_rfm.png")
plt.close()

# Salvar os dados RFM para uso posterior
rfm.to_csv('../dados_rfm.csv', index=False)
print("Dados RFM salvos em 'dados_rfm.csv'")

print("\n# Conclusões da Análise Exploratória")
print("""
Esta análise exploratória nos permitiu entender melhor os dados da ClickBus e identificar padrões importantes:

1. Comportamento Temporal: Identificamos padrões sazonais nas compras ao longo do ano e da semana.

2. Padrões de Rotas: Visualizamos as rotas mais populares e os destinos mais procurados pelos clientes.

3. Perfil de Clientes: Analisamos a distribuição de compras e gastos por cliente, preparando o terreno para a segmentação.

4. Métricas RFM: Calculamos as métricas de Recência, Frequência e Valor Monetário, que serão a base para a segmentação de clientes.

No próximo script, realizaremos a segmentação dos clientes utilizando as métricas RFM e o algoritmo K-means.
""") 
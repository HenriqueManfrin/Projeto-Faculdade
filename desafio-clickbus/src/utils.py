import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

def save_model(model, filename):
    """
    Salva um modelo treinado em um arquivo
    
    Args:
        model: Modelo a ser salvo
        filename: Nome do arquivo
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Modelo salvo em {filename}")

def load_model(filename):
    """
    Carrega um modelo treinado de um arquivo
    
    Args:
        filename: Nome do arquivo
        
    Returns:
        Modelo carregado
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Modelo carregado de {filename}")
    return model

def create_sample_data(df, sample_size=10000, random_state=42):
    """
    Cria uma amostra dos dados para testes e desenvolvimento
    
    Args:
        df: DataFrame original
        sample_size: Tamanho da amostra
        random_state: Semente aleatória
        
    Returns:
        DataFrame com a amostra
    """
    # Garantir que temos clientes completos (todas as compras deles)
    unique_customers = df['customer_id'].unique()
    
    # Selecionar uma amostra aleatória de clientes
    np.random.seed(random_state)
    sampled_customers = np.random.choice(unique_customers, 
                                        size=min(sample_size, len(unique_customers)), 
                                        replace=False)
    
    # Filtrar o DataFrame para incluir apenas os clientes selecionados
    sample_df = df[df['customer_id'].isin(sampled_customers)]
    
    print(f"Amostra criada com {len(sample_df)} registros de {len(sampled_customers)} clientes")
    
    return sample_df

def plot_purchase_patterns(df):
    """
    Plota padrões de compra ao longo do tempo
    
    Args:
        df: DataFrame com os dados de compras
    """
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Compras por mês
    plt.subplot(2, 2, 1)
    df['month'] = df['booking_date'].dt.month
    monthly_purchases = df.groupby('month').size()
    monthly_purchases.plot(kind='bar', color='skyblue')
    plt.title('Compras por Mês', fontsize=14)
    plt.xlabel('Mês')
    plt.ylabel('Número de Compras')
    
    # Plot 2: Compras por dia da semana
    plt.subplot(2, 2, 2)
    df['dayofweek'] = df['booking_date'].dt.dayofweek
    day_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    weekday_purchases = df.groupby('dayofweek').size()
    weekday_purchases.index = [day_names[i] for i in weekday_purchases.index]
    weekday_purchases.plot(kind='bar', color='salmon')
    plt.title('Compras por Dia da Semana', fontsize=14)
    plt.xlabel('Dia da Semana')
    plt.ylabel('Número de Compras')
    
    # Plot 3: Histograma de valores de compra
    plt.subplot(2, 2, 3)
    plt.hist(df['price'], bins=30, color='lightgreen')
    plt.title('Distribuição de Valores de Compra', fontsize=14)
    plt.xlabel('Valor (R$)')
    plt.ylabel('Frequência')
    
    # Plot 4: Antecedência da compra (dias antes do embarque)
    plt.subplot(2, 2, 4)
    advance_days = (df['boarding_date'] - df['booking_date']).dt.days
    plt.hist(advance_days, bins=30, color='lightpink')
    plt.title('Antecedência da Compra', fontsize=14)
    plt.xlabel('Dias antes do embarque')
    plt.ylabel('Frequência')
    
    plt.tight_layout()
    plt.savefig('../img/exploracao/padroes_compra.png')

def plot_route_analysis(df):
    """
    Plota análises de rotas
    
    Args:
        df: DataFrame com os dados de compras
    """
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Top 10 rotas mais populares
    plt.subplot(2, 2, 1)
    route_counts = df.groupby('route').size().reset_index(name='count')
    top_routes = route_counts.sort_values('count', ascending=False).head(10)
    top_routes.plot(kind='barh', x='route', y='count', color='skyblue')
    plt.title('Top 10 Rotas Mais Populares', fontsize=14)
    plt.xlabel('Número de Compras')
    plt.ylabel('Rota (Origem-Destino)')
    
    # Plot 2: Top 10 origens mais comuns
    plt.subplot(2, 2, 2)
    origin_counts = df.groupby('origin_id').size().reset_index(name='count')
    top_origins = origin_counts.sort_values('count', ascending=False).head(10)
    top_origins.plot(kind='barh', x='origin_id', y='count', color='salmon')
    plt.title('Top 10 Origens Mais Comuns', fontsize=14)
    plt.xlabel('Número de Compras')
    plt.ylabel('ID da Origem')
    
    # Plot 3: Top 10 destinos mais comuns
    plt.subplot(2, 2, 3)
    dest_counts = df.groupby('destination_id').size().reset_index(name='count')
    top_dests = dest_counts.sort_values('count', ascending=False).head(10)
    top_dests.plot(kind='barh', x='destination_id', y='count', color='lightgreen')
    plt.title('Top 10 Destinos Mais Comuns', fontsize=14)
    plt.xlabel('Número de Compras')
    plt.ylabel('ID do Destino')
    
    # Plot 4: Preço médio por rota (top 10 rotas)
    plt.subplot(2, 2, 4)
    route_price = df.groupby('route')['price'].mean().reset_index()
    top_price_routes = route_price.sort_values('price', ascending=False).head(10)
    top_price_routes.plot(kind='barh', x='route', y='price', color='lightpink')
    plt.title('Top 10 Rotas Mais Caras (Preço Médio)', fontsize=14)
    plt.xlabel('Preço Médio (R$)')
    plt.ylabel('Rota (Origem-Destino)')
    
    plt.tight_layout()
    plt.savefig('../img/exploracao/analise_rotas.png')

def create_summary_report(segment_data, purchase_results, route_results):
    """
    Cria um relatório resumido dos resultados
    
    Args:
        segment_data: Dados de segmentação
        purchase_results: Resultados do modelo de previsão de compra
        route_results: Resultados do modelo de previsão de rota
        
    Returns:
        String com o relatório
    """
    today = datetime.now().strftime("%d/%m/%Y")
    
    report = f"""
    RELATÓRIO DE ANÁLISE - CLICKBUS
    Data: {today}
    
    1. SEGMENTAÇÃO DE CLIENTES
    -------------------------
    Número de segmentos: {segment_data['segment'].nunique()}
    
    Distribuição de clientes por segmento:
    {segment_data.groupby('segment_name').size().reset_index(name='count').to_string(index=False)}
    
    2. PREVISÃO DE COMPRAS
    ---------------------
    Acurácia do modelo: {purchase_results['accuracy']:.2f}
    F1-Score: {purchase_results['f1']:.2f}
    
    Features mais importantes:
    {purchase_results['importance'].head(5).to_string(index=False)}
    
    3. PREVISÃO DE ROTAS
    -------------------
    Acurácia do modelo: {route_results['accuracy']:.2f}
    
    Features mais importantes:
    {route_results['importance'].head(5).to_string(index=False)}
    
    CONCLUSÃO
    ---------
    A análise identificou segmentos distintos de clientes com base no seu comportamento de compra.
    O modelo de previsão de compras consegue identificar clientes com maior probabilidade de realizar uma nova compra.
    O modelo de recomendação de rotas pode ajudar a direcionar ofertas específicas para cada cliente.
    """
    
    # Salvar relatório em arquivo
    with open('relatorio_analise.txt', 'w') as file:
        file.write(report)
    
    return report 
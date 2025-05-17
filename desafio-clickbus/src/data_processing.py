import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

def load_data(file_path):
    """
    Carrega os dados do arquivo CSV e realiza transformações iniciais.
    
    Args:
        file_path: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados processados
    """
    print("Carregando dados...")
    # Carregando apenas as colunas necessárias para reduzir o uso de memória
    df = pd.read_csv(file_path, parse_dates=['booking_date', 'boarding_date'])
    
    # Converter tipos de dados para reduzir uso de memória
    df['booking_id'] = df['booking_id'].astype('int32')
    df['customer_id'] = df['customer_id'].astype('int32')
    df['origin_id'] = df['origin_id'].astype('int16')
    df['destination_id'] = df['destination_id'].astype('int16')
    
    print(f"Dados carregados com sucesso. Dimensões: {df.shape}")
    return df

def prepare_rfm_data(df, reference_date=None):
    """
    Prepara os dados para análise RFM (Recency, Frequency, Monetary)
    
    Args:
        df: DataFrame com os dados de compras
        reference_date: Data de referência para cálculo da recência (se None, usa a data máxima + 1 dia)
        
    Returns:
        DataFrame com os dados RFM por cliente
    """
    if reference_date is None:
        reference_date = df['booking_date'].max() + timedelta(days=1)
    
    # Agrupamento por cliente para calcular RFM
    rfm = df.groupby('customer_id').agg({
        'booking_date': lambda x: (reference_date - x.max()).days,  # Recência (R)
        'booking_id': 'nunique',  # Frequência (F)
        'price': 'sum'  # Valor Monetário (M)
    }).reset_index()
    
    # Renomear colunas
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    return rfm

def create_time_features(df):
    """
    Cria features temporais a partir das datas de compra e embarque
    
    Args:
        df: DataFrame com os dados de compras
        
    Returns:
        DataFrame com as features temporais adicionadas
    """
    # Extrair componentes da data
    df['booking_year'] = df['booking_date'].dt.year
    df['booking_month'] = df['booking_date'].dt.month
    df['booking_day'] = df['booking_date'].dt.day
    df['booking_dayofweek'] = df['booking_date'].dt.dayofweek
    df['booking_weekend'] = df['booking_dayofweek'].isin([5, 6]).astype(int)
    
    # Extrair componentes da data de embarque
    df['boarding_year'] = df['boarding_date'].dt.year
    df['boarding_month'] = df['boarding_date'].dt.month
    df['boarding_day'] = df['boarding_date'].dt.day
    df['boarding_dayofweek'] = df['boarding_date'].dt.dayofweek
    df['boarding_weekend'] = df['boarding_dayofweek'].isin([5, 6]).astype(int)
    
    # Calcular dias de antecedência da compra
    df['days_before_boarding'] = (df['boarding_date'] - df['booking_date']).dt.days
    
    return df

def create_purchase_features(df):
    """
    Cria features relacionadas ao histórico de compras
    
    Args:
        df: DataFrame com os dados de compras
        
    Returns:
        DataFrame com as features de compra adicionadas
    """
    # Ordenar por cliente e data
    df = df.sort_values(['customer_id', 'booking_date'])
    
    # Calcular tempo desde a última compra
    df['prev_purchase'] = df.groupby('customer_id')['booking_date'].shift(1)
    df['days_since_prev_purchase'] = (df['booking_date'] - df['prev_purchase']).dt.days
    
    # Calcular média de dias entre compras por cliente
    avg_days = df.groupby('customer_id')['days_since_prev_purchase'].mean().reset_index()
    avg_days.columns = ['customer_id', 'avg_days_between_purchases']
    
    # Mesclar de volta ao DataFrame original
    df = pd.merge(df, avg_days, on='customer_id', how='left')
    
    # Calcular contagem de compras prévias
    df['purchase_count'] = df.groupby('customer_id').cumcount() + 1
    
    return df

def create_route_features(df):
    """
    Cria features relacionadas a rotas e destinos
    
    Args:
        df: DataFrame com os dados de compras
        
    Returns:
        DataFrame com as features de rota adicionadas
    """
    # Criar identificador de rota (origem-destino)
    df['route'] = df['origin_id'].astype(str) + '-' + df['destination_id'].astype(str)
    
    # Calcular frequência de rotas por cliente
    route_freq = df.groupby(['customer_id', 'route']).size().reset_index(name='route_frequency')
    
    # Encontrar rota mais frequente por cliente
    most_frequent_route = route_freq.sort_values(['customer_id', 'route_frequency'], ascending=[True, False])
    most_frequent_route = most_frequent_route.groupby('customer_id').first().reset_index()
    most_frequent_route.columns = ['customer_id', 'most_frequent_route', 'most_frequent_route_count']
    
    # Mesclar de volta ao DataFrame original
    df = pd.merge(df, most_frequent_route[['customer_id', 'most_frequent_route']], on='customer_id', how='left')
    
    # Indicador se a compra atual é da rota mais frequente
    df['is_favorite_route'] = (df['route'] == df['most_frequent_route']).astype(int)
    
    return df

def prepare_for_next_purchase_prediction(df, future_days=30):
    """
    Prepara os dados para prever se um cliente fará uma compra nos próximos X dias
    
    Args:
        df: DataFrame com os dados de compras
        future_days: Número de dias para considerar no futuro
        
    Returns:
        X: Features para o modelo
        y: Target (comprou em X dias = 1, caso contrário = 0)
    """
    # Ordenar por cliente e data
    df = df.sort_values(['customer_id', 'booking_date'])
    
    # Para cada compra, verificar se o cliente fez outra compra nos próximos X dias
    customers = df['customer_id'].unique()
    results = []
    
    print(f"Preparando dados para previsão de compra nos próximos {future_days} dias...")
    for customer in tqdm(customers):
        customer_purchases = df[df['customer_id'] == customer].copy()
        for idx, row in customer_purchases.iterrows():
            current_date = row['booking_date']
            future_date = current_date + timedelta(days=future_days)
            
            # Verificar se há uma compra nos próximos X dias
            next_purchases = customer_purchases[
                (customer_purchases['booking_date'] > current_date) & 
                (customer_purchases['booking_date'] <= future_date)
            ]
            
            bought_in_future = 1 if len(next_purchases) > 0 else 0
            
            # Adicionar indicador ao resultado
            result_row = row.copy()
            result_row['bought_in_future_days'] = bought_in_future
            results.append(result_row)
    
    result_df = pd.DataFrame(results)
    
    # Selecionar features relevantes
    features = [
        'customer_id', 'recency', 'frequency', 'monetary',
        'days_since_prev_purchase', 'avg_days_between_purchases',
        'purchase_count', 'is_favorite_route', 'price',
        'days_before_boarding', 'booking_weekend', 'boarding_weekend'
    ]
    
    # Subset de features que estão disponíveis
    available_features = [f for f in features if f in result_df.columns]
    
    X = result_df[available_features]
    y = result_df['bought_in_future_days']
    
    return X, y

def prepare_for_next_route_prediction(df):
    """
    Prepara os dados para prever qual será a próxima rota de um cliente
    
    Args:
        df: DataFrame com os dados de compras
        
    Returns:
        X: Features para o modelo
        y: Target (próxima rota)
    """
    # Ordenar por cliente e data
    df_sorted = df.sort_values(['customer_id', 'booking_date'])
    
    # Para cada compra, determinar qual será a próxima rota do cliente
    df_sorted['next_route'] = df_sorted.groupby('customer_id')['route'].shift(-1)
    
    # Remover registros sem próxima rota (última compra do cliente)
    df_with_next = df_sorted.dropna(subset=['next_route'])
    
    # Selecionar features relevantes
    features = [
        'customer_id', 'recency', 'frequency', 'monetary',
        'days_since_prev_purchase', 'avg_days_between_purchases',
        'purchase_count', 'is_favorite_route', 'price',
        'days_before_boarding', 'booking_weekend', 'boarding_weekend',
        'route', 'origin_id', 'destination_id'
    ]
    
    # Subset de features que estão disponíveis
    available_features = [f for f in features if f in df_with_next.columns]
    
    X = df_with_next[available_features]
    y = df_with_next['next_route']
    
    return X, y 
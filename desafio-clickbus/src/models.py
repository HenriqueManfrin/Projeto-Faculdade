import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def perform_rfm_segmentation(rfm_data, n_clusters=4, random_state=42):
    """
    Realiza a segmentação RFM usando K-means
    
    Args:
        rfm_data: DataFrame com os dados RFM
        n_clusters: Número de clusters (segmentos)
        random_state: Semente aleatória
        
    Returns:
        DataFrame com os dados RFM e o segmento de cada cliente
    """
    # Cópia dos dados
    rfm = rfm_data.copy()
    
    # Selecionar apenas as colunas RFM para clustering
    X = rfm[['recency', 'frequency', 'monetary']]
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm['segment'] = kmeans.fit_predict(X_scaled)
    
    # Avaliar qualidade dos clusters
    silhouette = silhouette_score(X_scaled, rfm['segment'])
    print(f"Silhouette Score: {silhouette:.4f}")
    
    # Analisar características de cada segmento
    segment_analysis = rfm.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).reset_index()
    
    segment_analysis['customer_percentage'] = segment_analysis['customer_id'] / segment_analysis['customer_id'].sum() * 100
    segment_analysis.rename(columns={'customer_id': 'customers_count'}, inplace=True)
    
    print("\nCaracterísticas dos Segmentos:")
    print(segment_analysis)
    
    # Nomear os segmentos com base nas características
    segment_names = []
    for _, row in segment_analysis.iterrows():
        recency = row['recency']
        frequency = row['frequency']
        monetary = row['monetary']
        
        if recency < segment_analysis['recency'].median():
            recency_label = "Recente"
        else:
            recency_label = "Não Recente"
            
        if frequency > segment_analysis['frequency'].median():
            frequency_label = "Frequente"
        else:
            frequency_label = "Não Frequente"
            
        if monetary > segment_analysis['monetary'].median():
            monetary_label = "Alto Valor"
        else:
            monetary_label = "Baixo Valor"
            
        segment_names.append(f"{recency_label}, {frequency_label}, {monetary_label}")
    
    # Criar mapeamento de segmentos para nomes
    segment_mapping = {i: name for i, name in enumerate(segment_names)}
    
    # Adicionar nomes dos segmentos ao DataFrame original
    rfm['segment_name'] = rfm['segment'].map(segment_mapping)
    
    return rfm, segment_mapping

def train_next_purchase_model(X, y, test_size=0.2, random_state=42):
    """
    Treina um modelo para prever se um cliente fará uma compra em breve
    
    Args:
        X: Features
        y: Target (1 = compra nos próximos X dias, 0 = não compra)
        test_size: Proporção de dados para teste
        random_state: Semente aleatória
        
    Returns:
        Modelo treinado e métricas de avaliação
    """
    # Remover colunas não úteis para o modelo
    X_model = X.drop('customer_id', axis=1, errors='ignore')
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size, random_state=random_state)
    
    # Treinar modelo (Random Forest)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nResultados do Modelo de Previsão de Compra:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Features mais importantes
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeatures mais importantes:")
    print(feature_importance.head(10))
    
    return model, {'model': model, 'accuracy': accuracy, 'f1': f1, 'importance': feature_importance}

def train_next_route_model(X, y, test_size=0.2, random_state=42):
    """
    Treina um modelo para prever qual será o próximo trecho de um cliente
    
    Args:
        X: Features
        y: Target (rota no formato 'origem-destino')
        test_size: Proporção de dados para teste
        random_state: Semente aleatória
        
    Returns:
        Modelo treinado e métricas de avaliação
    """
    # Remover colunas não úteis para o modelo
    X_model = X.drop(['customer_id', 'route', 'origin_id', 'destination_id'], axis=1, errors='ignore')
    
    # Codificar a variável alvo (rota)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_encoded, test_size=test_size, random_state=random_state)
    
    # Treinar modelo (Decision Tree para multiclasse)
    model = DecisionTreeClassifier(max_depth=15, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Avaliar modelo
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nResultados do Modelo de Previsão de Rota:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Features mais importantes
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeatures mais importantes:")
    print(feature_importance.head(10))
    
    return model, label_encoder, {'model': model, 'encoder': label_encoder, 'accuracy': accuracy, 'importance': feature_importance}

def visualize_segments(rfm_data, segment_mapping):
    """
    Visualiza os segmentos de clientes obtidos
    
    Args:
        rfm_data: DataFrame com os dados RFM e segmentos
        segment_mapping: Mapeamento de números de segmentos para nomes
    """
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Distribuição de segmentos
    plt.subplot(2, 2, 1)
    segments_count = rfm_data['segment'].value_counts().sort_index()
    segments_count.index = [segment_mapping[i] for i in segments_count.index]
    
    ax = segments_count.plot(kind='bar', color='skyblue')
    plt.title('Distribuição de Clientes por Segmento', fontsize=14)
    plt.xlabel('Segmento')
    plt.ylabel('Número de Clientes')
    plt.xticks(rotation=90)
    
    # Plot 2: Recência vs Frequência
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(rfm_data['recency'], rfm_data['frequency'], 
              c=rfm_data['segment'], cmap='viridis', alpha=0.6)
    plt.title('Recência vs Frequência por Segmento', fontsize=14)
    plt.xlabel('Recência (dias)')
    plt.ylabel('Frequência (número de compras)')
    plt.colorbar(scatter, label='Segmento')
    
    # Plot 3: Frequência vs Valor
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(rfm_data['frequency'], rfm_data['monetary'], 
              c=rfm_data['segment'], cmap='viridis', alpha=0.6)
    plt.title('Frequência vs Valor por Segmento', fontsize=14)
    plt.xlabel('Frequência (número de compras)')
    plt.ylabel('Valor Monetário Total (R$)')
    plt.colorbar(scatter, label='Segmento')
    
    # Plot 4: Recência vs Valor
    plt.subplot(2, 2, 4)
    scatter = plt.scatter(rfm_data['recency'], rfm_data['monetary'], 
              c=rfm_data['segment'], cmap='viridis', alpha=0.6)
    plt.title('Recência vs Valor Monetário por Segmento', fontsize=14)
    plt.xlabel('Recência (dias)')
    plt.ylabel('Valor Monetário Total (R$)')
    plt.colorbar(scatter, label='Segmento')
    
    plt.tight_layout()
    plt.savefig('../img/segmentacao/segmentos.png')
    
def predict_purchases(model, customer_data):
    """
    Prevê quais clientes farão uma compra nos próximos dias
    
    Args:
        model: Modelo treinado
        customer_data: Dados dos clientes para previsão
        
    Returns:
        DataFrame com as previsões
    """
    # Remover colunas não úteis para o modelo
    X_pred = customer_data.drop('customer_id', axis=1, errors='ignore')
    
    # Fazer previsões
    predictions = model.predict(X_pred)
    probabilities = model.predict_proba(X_pred)[:, 1]
    
    # Criar DataFrame de resultados
    results = pd.DataFrame({
        'customer_id': customer_data['customer_id'],
        'will_purchase': predictions,
        'purchase_probability': probabilities
    })
    
    return results

def predict_routes(model, label_encoder, customer_data):
    """
    Prevê qual será o próximo trecho de um cliente
    
    Args:
        model: Modelo treinado
        label_encoder: Encoder das rotas
        customer_data: Dados dos clientes para previsão
        
    Returns:
        DataFrame com as previsões
    """
    # Remover colunas não úteis para o modelo
    X_pred = customer_data.drop(['customer_id', 'route', 'origin_id', 'destination_id'], axis=1, errors='ignore')
    
    # Fazer previsões
    pred_encoded = model.predict(X_pred)
    predictions = label_encoder.inverse_transform(pred_encoded)
    
    # Criar DataFrame de resultados
    results = pd.DataFrame({
        'customer_id': customer_data['customer_id'],
        'predicted_route': predictions
    })
    
    return results 
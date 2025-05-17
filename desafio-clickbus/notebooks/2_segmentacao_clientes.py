#!/usr/bin/env python
# coding: utf-8

# # Segmentação de Clientes da ClickBus
# 
# Este script implementa a segmentação de clientes utilizando a técnica RFM (Recência, Frequência, Valor Monetário) e o algoritmo K-means.

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locais
import sys
import os

# Adicionar diretório pai ao sys.path para encontrar os módulos src
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from src.data_processing import load_data, prepare_rfm_data
from src.models import perform_rfm_segmentation, visualize_segments
from src.utils import create_sample_data

# Configurações de visualização
plt.style.use('ggplot')
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)

print("# Carregamento e Preparação dos Dados")
print("Vamos carregar os dados e calcular as métricas RFM necessárias para a segmentação.")

# Tentar carregar dados RFM calculados anteriormente
try:
    print("Tentando carregar dados RFM pré-processados...")
    rfm_data = pd.read_csv('../dados_rfm.csv')
    print(f"Dados RFM carregados com sucesso. Dimensões: {rfm_data.shape}")
except:
    print("Dados RFM não encontrados. Carregando dados brutos...")
    
    # Carregar o arquivo de dados original
    file_path = '../dados_desafio_fiap/hash/df_t.csv'
    
    try:
        # Tente carregar o arquivo real
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
    
    # Preparar dados RFM
    reference_date = df['booking_date'].max() + timedelta(days=1)
    rfm_data = prepare_rfm_data(df, reference_date=reference_date)

# Visualizar os dados RFM preparados
print("\nPrimeiras linhas dos dados RFM:")
print(rfm_data.head())

print("\n# Determinação do Número Ideal de Clusters")
print("Vamos utilizar o método do cotovelo e o coeficiente de silhueta para determinar o número ideal de segmentos.")

# Selecionar apenas as variáveis RFM para clustering
X = rfm_data[['recency', 'frequency', 'monetary']]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do cotovelo para determinar o número ideal de clusters
inertia = []
silhouette_scores = []
range_n_clusters = range(2, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
    # Calcular o coeficiente de silhueta
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)
    
    print(f"Para n_clusters = {n_clusters}, o coeficiente de silhueta é {silhouette_avg:.3f}")

# Plotar o método do cotovelo
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, inertia, 'o-', markersize=8)
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, silhouette_scores, 'o-', markersize=8)
plt.title('Coeficiente de Silhueta')
plt.xlabel('Número de Clusters')
plt.ylabel('Coeficiente de Silhueta')
plt.grid(True)

plt.tight_layout()
plt.savefig('../img/segmentacao/numero_ideal_clusters.png')
print("Gráfico salvo: img/segmentacao/numero_ideal_clusters.png")
plt.close()

print("\nCom base nos resultados acima, selecionamos o número ideal de clusters.")
print("Para a ClickBus, vamos utilizar 4 segmentos, que é um número comum em análises RFM e permite uma interpretação clara.")

print("\n# Segmentação com K-means")
print("Agora vamos aplicar o algoritmo K-means para segmentar os clientes em 4 grupos distintos.")

# Aplicar K-means com 4 clusters
rfm_segmented, segment_mapping = perform_rfm_segmentation(rfm_data, n_clusters=4)

# Visualizar os primeiros registros com os segmentos atribuídos
print("\nPrimeiros registros com os segmentos atribuídos:")
print(rfm_segmented.head())

print("\n# Análise dos Segmentos")
print("Vamos analisar as características de cada segmento para entender melhor o perfil dos clientes.")

# Análise detalhada dos segmentos
segment_analysis = rfm_segmented.groupby('segment_name').agg({
    'recency': ['mean', 'median', 'min', 'max'],
    'frequency': ['mean', 'median', 'min', 'max'],
    'monetary': ['mean', 'median', 'min', 'max'],
    'customer_id': 'count'
})

# Calcular o percentual de clientes em cada segmento
segment_analysis[('customer_id', 'percentage')] = segment_analysis[('customer_id', 'count')] / segment_analysis[('customer_id', 'count')].sum() * 100

print("\nAnálise detalhada dos segmentos:")
print(segment_analysis)

# Visualizar a distribuição dos segmentos
plt.figure(figsize=(10, 6))
segment_counts = rfm_segmented['segment_name'].value_counts()
segment_counts.plot(kind='bar', color='skyblue')
plt.title('Distribuição de Clientes por Segmento')
plt.xlabel('Segmento')
plt.ylabel('Número de Clientes')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('../img/segmentacao/distribuicao_segmentos.png')
print("Gráfico salvo: img/segmentacao/distribuicao_segmentos.png")
plt.close()

# Visualizar os segmentos em gráficos de dispersão
visualize_segments(rfm_segmented, segment_mapping)
print("Gráfico salvo: img/segmentacao/segmentos.png")

# Visualização 3D dos segmentos (salvar como HTML para interatividade)
fig = px.scatter_3d(rfm_segmented, x='recency', y='frequency', z='monetary',
                   color='segment_name',
                   title='Visualização 3D dos Segmentos de Clientes',
                   labels={'recency': 'Recência (dias)', 'frequency': 'Frequência (compras)', 'monetary': 'Valor (R$)'})
fig.write_html('../segmentos_3d.html')
print("Visualização 3D salva: segmentos_3d.html")

print("\n# Interpretação dos Segmentos")
print("Vamos interpretar cada segmento e definir estratégias de marketing específicas para cada grupo.")

# Criar um DataFrame com a interpretação dos segmentos
segment_interpretation = pd.DataFrame({
    'Segmento': list(segment_mapping.values()),
    'Descrição': [
        'Clientes que compram com frequência e gastam valores elevados, mas não fizeram compras recentemente',
        'Clientes que fizeram compras recentemente, mas com baixa frequência e valor',
        'Clientes que não compram há muito tempo, com baixa frequência e valor',
        'Clientes que compram com frequência, gastam valores elevados e fizeram compras recentemente'
    ],
    'Estratégia Recomendada': [
        'Campanha de reativação com ofertas personalizadas baseadas nas preferências anteriores',
        'Incentivos para segunda compra, com foco em aumentar o ticket médio',
        'Grandes descontos ou promoções agressivas para reconquistar',
        'Programa de fidelidade, antecipação de ofertas premium, trechos exclusivos'
    ],
    'Canais': [
        'Email marketing, SMS, notificações push',
        'Email marketing, redes sociais',
        'Email marketing com grandes ofertas, remarketing',
        'Email marketing personalizado, app, atendimento diferenciado'
    ],
    'Nome Comercial': [
        'Viajantes Frequentes em Pausa',
        'Novos Exploradores',
        'Viajantes Distantes',
        'Viajantes VIP'
    ]
})

print("\nNomes e estratégias para cada segmento:")
for _, row in segment_interpretation.iterrows():
    print(f"\nSegmento: {row['Segmento']}")
    print(f"Nome Comercial: {row['Nome Comercial']}")
    print(f"Descrição: {row['Descrição']}")
    print(f"Estratégia: {row['Estratégia Recomendada']}")
    print(f"Canais: {row['Canais']}")

print("\n# Salvar os Resultados")
print("Vamos salvar os resultados da segmentação para uso posterior nos modelos de previsão.")

# Salvar os resultados em um arquivo CSV
rfm_segmented.to_csv('../segmentacao_clientes.csv', index=False)
print("Resultados da segmentação salvos em 'segmentacao_clientes.csv'")

print("\n# Conclusões")
print("""
Neste script, realizamos a segmentação dos clientes da ClickBus utilizando a técnica RFM e o algoritmo K-means. Identificamos 4 segmentos distintos de clientes, cada um com características e necessidades específicas:

1. Viajantes Frequentes em Pausa: Clientes que compram com frequência e gastam valores elevados, mas não fizeram compras recentemente. Estes clientes são valiosos e devem ser reativados com ofertas personalizadas.

2. Novos Exploradores: Clientes que fizeram compras recentemente, mas com baixa frequência e valor. O foco aqui é aumentar a frequência e o valor das compras.

3. Viajantes Distantes: Clientes que não compram há muito tempo, com baixa frequência e valor. Este segmento precisa de incentivos fortes para retornar.

4. Viajantes VIP: Clientes que compram com frequência, gastam valores elevados e fizeram compras recentemente. São os clientes mais valiosos e devem receber tratamento premium.

Essa segmentação permite à ClickBus direcionar melhor suas estratégias de marketing, oferecendo experiências e promoções personalizadas para cada grupo de clientes, aumentando assim a eficiência de suas ações e a satisfação dos clientes.

No próximo script, utilizaremos essa segmentação como input para o modelo de previsão da próxima compra.
""") 
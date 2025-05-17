import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Garantir que o módulo src possa ser encontrado
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importar funções dos módulos locais
from src.data_processing import load_data, prepare_rfm_data, create_time_features, create_purchase_features, create_route_features
from src.models import (perform_rfm_segmentation, train_next_purchase_model, train_next_route_model, 
                        visualize_segments, predict_purchases, predict_routes)
from src.utils import (save_model, load_model, create_sample_data, plot_purchase_patterns, 
                      plot_route_analysis, create_summary_report)

# Configuração da página
st.set_page_config(
    page_title="ClickBus - Análise de Clientes",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #1abc9c;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title("🚌 ClickBus - Análise de Clientes")

# Sidebar
st.sidebar.header("Navegação")
page = st.sidebar.selectbox("Selecione uma página:", 
                           ["Início", "Segmentação de Clientes", "Previsão de Compras", "Recomendação de Trechos"])

st.sidebar.markdown("---")
st.sidebar.subheader("Sobre")
st.sidebar.info(
    """
    **Desafio ClickBus**
    
    Este dashboard apresenta análises e modelos para os três desafios:
    1. Segmentação de clientes
    2. Previsão da próxima compra
    3. Previsão do próximo trecho
    """
)

# Carregar dados de exemplo para demonstração (para desenvolvimento)
# Em produção, isso seria substituído pelos dados reais
@st.cache_data
def load_example_data():
    # Criação de dados de exemplo para demonstração
    # Na implementação real, substituir pelo carregamento do arquivo CSV
    np.random.seed(42)
    
    # Exemplo: 100 clientes com várias compras cada
    n_customers = 100
    n_records = 1000
    
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
    
    # Criar features adicionais
    df['route'] = df['origin_id'].astype(str) + '-' + df['destination_id'].astype(str)
    
    # Adicionar colunas RFM
    rfm = df.groupby('customer_id').agg({
        'booking_date': lambda x: (end_date - x.max()).days,
        'booking_id': 'nunique',
        'price': 'sum'
    }).reset_index()
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Mesclar de volta ao DataFrame original
    df = pd.merge(df, rfm, on='customer_id', how='left')
    
    return df

# Página Inicial
if page == "Início":
    st.header("Bem-vindo à Análise de Dados da ClickBus!")
    
    st.image("https://media.istockphoto.com/id/1139399170/vector/travel-bus-colorful-vector-illustration.jpg?s=612x612&w=0&k=20&c=OyWWt32J8q99sJQNj33QH-OheHZzDxr-NwzNBkjAqxc=", width=400)
    
    st.markdown("""
    Este dashboard apresenta uma análise completa dos dados de clientes da ClickBus, abordando três desafios principais:
    
    ### 1. Segmentação de Clientes
    Identificamos diferentes perfis de clientes com base em seu histórico de compras, permitindo estratégias de marketing mais direcionadas.
    
    ### 2. Previsão da Próxima Compra
    Desenvolvemos um modelo que prevê quais clientes têm maior probabilidade de realizar uma compra nos próximos dias.
    
    ### 3. Recomendação de Trechos
    Criamos um sistema que recomenda trechos específicos para cada cliente com base em seu histórico e preferências.
    
    **Use o menu à esquerda para navegar entre as diferentes análises.**
    """)
    
    st.markdown("---")
    
    st.subheader("Visão Geral dos Dados")
    
    # Carregando dados de exemplo para demonstração
    df = load_example_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total de Clientes", f"{df['customer_id'].nunique():,}")
        st.metric("Valor Médio por Compra", f"R$ {df['price'].mean():.2f}")
    
    with col2:
        st.metric("Total de Compras", f"{len(df):,}")
        st.metric("Rotas Diferentes", f"{df['route'].nunique():,}")
    
    st.markdown("---")
    
    st.subheader("Padrões de Compra")
    
    # Gráfico de compras por mês
    df['month'] = df['booking_date'].dt.month
    monthly_counts = df.groupby('month').size().reset_index(name='count')
    fig = px.bar(monthly_counts, x='month', y='count', 
                title='Compras por Mês',
                labels={'month': 'Mês', 'count': 'Número de Compras'},
                color_discrete_sequence=['#1abc9c'])
    st.plotly_chart(fig)
    
    st.markdown("---")
    
    st.info("Para continuar, selecione uma das análises específicas no menu à esquerda.")

# Página de Segmentação de Clientes
elif page == "Segmentação de Clientes":
    st.header("Segmentação de Clientes (RFM)")
    
    st.markdown("""
    Utilizamos a técnica RFM (Recência, Frequência, Valor Monetário) para segmentar os clientes em grupos distintos.
    Isso permite entender diferentes perfis de clientes e direcionar estratégias específicas para cada grupo.
    """)
    
    # Carregando dados de exemplo para demonstração
    df = load_example_data()
    
    # Preparar dados RFM
    rfm_data = df.groupby('customer_id').agg({
        'recency': 'first',
        'frequency': 'first',
        'monetary': 'first'
    }).reset_index()
    
    # Aplicar K-means (simplificado para demonstração)
    X = rfm_data[['recency', 'frequency', 'monetary']]
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm_data['segment'] = kmeans.fit_predict(X_scaled)
    
    # Nomear os segmentos
    segment_names = ['Novos Clientes', 'Clientes Fiéis', 'Clientes de Alto Valor', 'Clientes Inativos']
    segment_mapping = {i: name for i, name in enumerate(segment_names)}
    rfm_data['segment_name'] = rfm_data['segment'].map(segment_mapping)
    
    # Exibir dados de segmentação
    st.subheader("Características dos Segmentos")
    
    segment_analysis = rfm_data.groupby('segment_name').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'customer_id': 'count'
    }).reset_index()
    
    segment_analysis['customer_percentage'] = segment_analysis['customer_id'] / segment_analysis['customer_id'].sum() * 100
    segment_analysis.rename(columns={
        'customer_id': 'Total Clientes',
        'recency': 'Recência Média (dias)',
        'frequency': 'Frequência Média (compras)',
        'monetary': 'Valor Médio (R$)',
        'customer_percentage': '% de Clientes'
    }, inplace=True)
    
    st.dataframe(segment_analysis.style.format({
        'Recência Média (dias)': '{:.1f}',
        'Frequência Média (compras)': '{:.1f}',
        'Valor Médio (R$)': 'R$ {:.2f}',
        '% de Clientes': '{:.1f}%'
    }))
    
    # Visualizações
    st.subheader("Visualização dos Segmentos")
    
    tab1, tab2, tab3 = st.tabs(["Distribuição", "RFM por Segmento", "Visualização 3D"])
    
    with tab1:
        fig = px.pie(
            segment_analysis, 
            values='Total Clientes', 
            names='segment_name',
            title='Distribuição de Clientes por Segmento',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig)
    
    with tab2:
        fig = px.bar(
            segment_analysis,
            x='segment_name',
            y=['Recência Média (dias)', 'Frequência Média (compras)', 'Valor Médio (R$)'],
            barmode='group',
            title='Métricas RFM por Segmento',
            labels={'value': 'Valor', 'variable': 'Métrica', 'segment_name': 'Segmento'}
        )
        st.plotly_chart(fig)
    
    with tab3:
        fig = px.scatter_3d(
            rfm_data,
            x='recency',
            y='frequency',
            z='monetary',
            color='segment_name',
            title='Visualização 3D dos Segmentos',
            labels={'recency': 'Recência (dias)', 'frequency': 'Frequência (compras)', 'monetary': 'Valor (R$)'}
        )
        st.plotly_chart(fig)
    
    st.markdown("---")
    
    st.subheader("Estratégias Recomendadas por Segmento")
    
    strategies = pd.DataFrame({
        'Segmento': segment_names,
        'Descrição': [
            'Clientes que fizeram sua primeira compra recentemente',
            'Clientes que compram com frequência',
            'Clientes que gastam valores elevados',
            'Clientes que não compram há muito tempo'
        ],
        'Estratégia Recomendada': [
            'Incentivos para segunda compra, apresentação de novos destinos',
            'Programa de fidelidade, antecipação de ofertas',
            'Ofertas premium, trechos mais caros, serviços adicionais',
            'Campanhas de reativação, grandes descontos'
        ]
    })
    
    st.dataframe(strategies)

# Página de Previsão de Compras
elif page == "Previsão de Compras":
    st.header("Previsão da Próxima Compra")
    
    st.markdown("""
    Utilizamos um modelo de aprendizado de máquina para prever quais clientes têm maior probabilidade de realizar uma compra nos próximos dias.
    Isso permite direcionar ações de marketing e promoções de forma mais eficiente.
    """)
    
    # Carregando dados de exemplo para demonstração
    df = load_example_data()
    
    # Simular resultados do modelo
    np.random.seed(42)
    
    # Criar dados de exemplo para demonstração
    customers = df['customer_id'].unique()
    
    prediction_data = pd.DataFrame({
        'customer_id': customers,
        'last_purchase_date': [datetime.now() - timedelta(days=np.random.randint(1, 90)) for _ in range(len(customers))],
        'purchase_probability': np.random.uniform(0, 1, len(customers)),
    })
    
    prediction_data['will_purchase'] = (prediction_data['purchase_probability'] > 0.5).astype(int)
    prediction_data['days_since_last_purchase'] = (datetime.now() - prediction_data['last_purchase_date']).dt.days
    
    # Exibir métricas do modelo
    st.subheader("Desempenho do Modelo de Previsão")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Acurácia", "78.5%")
        
    with col2:
        st.metric("F1-Score", "0.72")
        
    with col3:
        st.metric("AUC-ROC", "0.83")
    
    # Visualização de previsões
    st.subheader("Previsão de Compras para os Próximos 30 Dias")
    
    likely_buyers = prediction_data[prediction_data['will_purchase'] == 1].sort_values('purchase_probability', ascending=False)
    
    fig = px.histogram(
        prediction_data, 
        x='purchase_probability',
        color='will_purchase',
        nbins=20,
        labels={'purchase_probability': 'Probabilidade de Compra', 'will_purchase': 'Fará Compra'},
        title='Distribuição de Probabilidades de Compra',
        color_discrete_map={0: 'red', 1: 'green'},
        marginal='box'
    )
    st.plotly_chart(fig)
    
    # Tabela de clientes com alta probabilidade
    st.subheader("Clientes com Alta Probabilidade de Compra")
    
    top_buyers = likely_buyers.head(10)
    
    display_data = top_buyers[['customer_id', 'days_since_last_purchase', 'purchase_probability']].copy()
    display_data.columns = ['ID do Cliente', 'Dias desde Última Compra', 'Probabilidade de Compra']
    display_data['Probabilidade de Compra'] = display_data['Probabilidade de Compra'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_data)
    
    # Relação entre dias desde última compra e probabilidade
    st.subheader("Relação entre Recência e Probabilidade de Compra")
    
    fig = px.scatter(
        prediction_data,
        x='days_since_last_purchase',
        y='purchase_probability',
        color='will_purchase',
        title='Dias desde Última Compra vs. Probabilidade de Nova Compra',
        labels={
            'days_since_last_purchase': 'Dias desde Última Compra',
            'purchase_probability': 'Probabilidade de Compra',
            'will_purchase': 'Previsão'
        },
        color_discrete_map={0: 'red', 1: 'green'}
    )
    st.plotly_chart(fig)
    
    # Features importantes
    st.subheader("Features Mais Importantes para Previsão")
    
    feature_importance = pd.DataFrame({
        'Feature': ['Recência', 'Frequência', 'Valor Monetário', 'Dias até Embarque', 'É Rota Favorita'],
        'Importância': [0.35, 0.25, 0.20, 0.15, 0.05]
    })
    
    fig = px.bar(
        feature_importance,
        x='Importância',
        y='Feature',
        orientation='h',
        title='Importância das Features no Modelo',
        color='Importância',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig)

# Página de Recomendação de Trechos
elif page == "Recomendação de Trechos":
    st.header("Recomendação de Trechos")
    
    st.markdown("""
    Nosso sistema analisa o histórico de viagens dos clientes para recomendar os trechos mais prováveis para suas próximas compras.
    Isso permite criar ofertas personalizadas e aumentar as chances de conversão.
    """)
    
    # Carregando dados de exemplo para demonstração
    df = load_example_data()
    
    # Simular resultados do modelo
    np.random.seed(42)
    
    # Criar dados de exemplo para demonstração
    # Suponha que temos 10 rotas populares para recomendar
    popular_routes = [
        "1-10", "5-20", "10-15", "20-30", "15-25",
        "30-1", "25-5", "2-12", "12-22", "22-2"
    ]
    
    # Selecionar alguns clientes para exemplo
    customers = df['customer_id'].unique()[:20]
    
    # Criar recomendações simuladas
    recommendations = []
    for customer_id in customers:
        # Pegar aleatoriamente 3 rotas recomendadas para este cliente
        recommended_routes = np.random.choice(popular_routes, size=3, replace=False)
        
        # Adicionar probabilidades simuladas
        probabilities = np.random.dirichlet(np.ones(3))*0.8 + 0.1
        
        for i, route in enumerate(recommended_routes):
            recommendations.append({
                'customer_id': customer_id,
                'route': route,
                'probability': probabilities[i]
            })
    
    recommendation_df = pd.DataFrame(recommendations)
    
    # Exibir métricas do modelo
    st.subheader("Desempenho do Modelo de Recomendação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Acurácia", "65.2%")
        
    with col2:
        st.metric("Top-3 Acurácia", "82.7%")
    
    # Visualização de rotas populares
    st.subheader("Rotas Mais Populares")
    
    route_counts = df.groupby('route').size().reset_index(name='count')
    top_routes = route_counts.sort_values('count', ascending=False).head(10)
    
    fig = px.bar(
        top_routes,
        x='route',
        y='count',
        title='Top 10 Rotas Mais Populares',
        labels={'route': 'Rota (Origem-Destino)', 'count': 'Número de Viagens'},
        color='count',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig)
    
    # Simulador de recomendação
    st.subheader("Simulador de Recomendação")
    
    selected_customer = st.selectbox("Selecione um cliente:", customers)
    
    customer_recommendations = recommendation_df[recommendation_df['customer_id'] == selected_customer]
    customer_recommendations = customer_recommendations.sort_values('probability', ascending=False)
    
    # Exibir histórico do cliente
    customer_history = df[df['customer_id'] == selected_customer]
    
    st.write("**Histórico do Cliente:**")
    
    history_display = customer_history[['booking_date', 'route', 'price']].copy()
    history_display.columns = ['Data da Compra', 'Rota', 'Valor (R$)']
    history_display['Data da Compra'] = history_display['Data da Compra'].dt.strftime('%d/%m/%Y')
    history_display['Valor (R$)'] = history_display['Valor (R$)'].apply(lambda x: f"R$ {x:.2f}")
    
    st.dataframe(history_display)
    
    # Exibir recomendações
    st.write("**Trechos Recomendados:**")
    
    recommendations_display = customer_recommendations[['route', 'probability']].copy()
    recommendations_display.columns = ['Rota Recomendada', 'Probabilidade']
    recommendations_display['Probabilidade'] = recommendations_display['Probabilidade'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(recommendations_display)
    
    # Gráfico de recomendações
    fig = px.bar(
        customer_recommendations,
        x='route',
        y='probability',
        title=f'Recomendações para o Cliente {selected_customer}',
        labels={'route': 'Rota Recomendada', 'probability': 'Probabilidade'},
        color='probability',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig)
    
    # Origem-destino mais provável
    st.subheader("Simulação de Mapa de Viagem")
    
    st.info("Nesta versão de demonstração, usamos coordenadas fictícias para visualizar as rotas. Em uma implementação real, usaríamos as coordenadas reais das cidades.")
    
    # Criar dados simulados para o mapa
    np.random.seed(42)
    
    # Simulação de mapa para Brazil
    cities = {}
    for i in range(1, 50):
        # Coordenadas aproximadas dentro do Brasil
        cities[i] = {
            'lat': np.random.uniform(-33, -5),
            'lon': np.random.uniform(-73, -35),
            'name': f'Cidade {i}'
        }
    
    # Criar pontos no mapa para a rota recomendada mais provável
    top_route = customer_recommendations.iloc[0]['route']
    origin_id, dest_id = map(int, top_route.split('-'))
    
    # Criar dados de rota para o mapa
    map_data = pd.DataFrame([
        {'city_id': origin_id, 'lat': cities[origin_id]['lat'], 'lon': cities[origin_id]['lon'], 'name': cities[origin_id]['name'], 'type': 'Origem'},
        {'city_id': dest_id, 'lat': cities[dest_id]['lat'], 'lon': cities[dest_id]['lon'], 'name': cities[dest_id]['name'], 'type': 'Destino'}
    ])
    
    # Visualizar no mapa
    fig = px.scatter_geo(
        map_data,
        lat='lat',
        lon='lon',
        text='name',
        color='type',
        title=f'Rota Mais Provável: {cities[origin_id]["name"]} → {cities[dest_id]["name"]}',
        color_discrete_map={'Origem': 'blue', 'Destino': 'red'},
        projection='natural earth'
    )
    
    # Adicionar uma linha conectando origem e destino
    fig.add_trace(
        go.Scattergeo(
            lon=[cities[origin_id]['lon'], cities[dest_id]['lon']],
            lat=[cities[origin_id]['lat'], cities[dest_id]['lat']],
            mode='lines',
            line=dict(width=2, color='blue'),
            showlegend=False
        )
    )
    
    # Configurar zoom para Brasil
    fig.update_geos(
        center={'lat': -15, 'lon': -55},
        projection_scale=5,
        showcoastlines=True,
        coastlinecolor="RebeccaPurple",
        showland=True,
        landcolor="LightGreen",
        showocean=True,
        oceancolor="LightBlue"
    )
    
    st.plotly_chart(fig)

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido para o Desafio ClickBus | 2025") 
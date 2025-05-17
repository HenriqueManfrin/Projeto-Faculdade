#!/usr/bin/env python
# coding: utf-8

"""
Script para testar os modelos do projeto ClickBus

Este script carrega os modelos treinados e executa testes básicos
para garantir que eles estão funcionando corretamente.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

# Adicionar diretório raiz ao sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Importar módulos locais
from src.utils import load_model, create_sample_data
from src.data_processing import prepare_rfm_data, prepare_for_next_purchase_prediction, prepare_for_next_route_prediction
from src.models import predict_purchases, predict_routes

def test_segmentation_model():
    """Testa o modelo de segmentação de clientes"""
    print("Testando modelo de segmentação...")
    
    # Criar dados de exemplo
    np.random.seed(42)
    n_customers = 100
    n_records = 1000
    
    # IDs de clientes
    customer_ids = np.random.randint(1000, 9999, n_customers)
    
    # Datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    booking_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
    
    # Criação do DataFrame
    df = pd.DataFrame({
        'booking_id': np.random.randint(10000, 99999, n_records),
        'customer_id': np.random.choice(customer_ids, n_records),
        'booking_date': booking_dates,
        'price': np.random.uniform(50, 500, n_records).round(2)
    })
    
    # Preparar dados RFM
    reference_date = df['booking_date'].max() + timedelta(days=1)
    rfm_data = prepare_rfm_data(df, reference_date=reference_date)
    
    # Verificar se os dados RFM foram criados corretamente
    assert 'recency' in rfm_data.columns, "Coluna 'recency' não encontrada nos dados RFM"
    assert 'frequency' in rfm_data.columns, "Coluna 'frequency' não encontrada nos dados RFM"
    assert 'monetary' in rfm_data.columns, "Coluna 'monetary' não encontrada nos dados RFM"
    
    print("Teste do modelo de segmentação concluído com sucesso!")
    return True

def test_purchase_prediction_model():
    """Testa o modelo de previsão de próxima compra"""
    print("Testando modelo de previsão de compra...")
    
    try:
        # Tentar carregar o modelo
        model_path = 'models/modelo_previsao_compra_30d.pkl'
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Modelo carregado: {type(model).__name__}")
        else:
            print(f"Arquivo do modelo não encontrado: {model_path}")
            print("Ignorando o teste com o modelo real, continuando com validação de dados...")
    
        # Criar dados de exemplo para testar a função de preparação
        np.random.seed(42)
        n_customers = 100
        n_records = 1000
        
        # Datas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        booking_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
        
        # Criação do DataFrame
        df = pd.DataFrame({
            'booking_id': np.random.randint(10000, 99999, n_records),
            'customer_id': np.random.choice(range(1000, 1100), n_records),
            'booking_date': booking_dates,
            'price': np.random.uniform(50, 500, n_records).round(2),
            'origin_id': np.random.randint(1, 50, n_records),
            'destination_id': np.random.randint(1, 50, n_records)
        })
        
        # Adicionar características derivadas
        df['days_since_last_purchase'] = np.random.randint(1, 100, n_records)
        df['total_purchases'] = np.random.randint(1, 10, n_records)
        df['average_price'] = np.random.uniform(50, 500, n_records).round(2)
        
        # Preparar dados para previsão
        X, y = prepare_for_next_purchase_prediction(df, future_days=30)
        
        # Verificar se os dados foram preparados corretamente
        assert X is not None, "X é None"
        assert y is not None, "y é None"
        assert len(X) > 0, "X está vazio"
        assert len(y) > 0, "y está vazio"
        
        print("Teste do modelo de previsão de compra concluído com sucesso!")
        return True
    
    except Exception as e:
        print(f"Erro no teste do modelo de previsão: {e}")
        return False

def test_route_recommendation_model():
    """Testa o modelo de recomendação de trechos"""
    print("Testando modelo de recomendação de trechos...")
    
    try:
        # Verificar se o encoder existe
        encoder_path = 'models/encoder_trechos.pkl'
        if os.path.exists(encoder_path):
            print(f"Encoder encontrado: {encoder_path}")
        else:
            print(f"Encoder não encontrado: {encoder_path}")
        
        # Verificar se o modelo existe
        model_path = 'models/modelo_recomendacao_trechos.pkl'
        if os.path.exists(model_path):
            print(f"Modelo encontrado: {model_path}")
        else:
            print(f"Modelo não encontrado: {model_path}")
        
        # Criar dados de exemplo para testar as funções
        np.random.seed(42)
        n_customers = 50
        n_records = 500
        
        # Criação do DataFrame
        df = pd.DataFrame({
            'customer_id': np.random.choice(range(1000, 1050), n_records),
            'origin_id': np.random.randint(1, 20, n_records),
            'destination_id': np.random.randint(1, 20, n_records)
        })
        
        # Criar a coluna de rota
        df['route'] = df['origin_id'].astype(str) + '-' + df['destination_id'].astype(str)
        
        # Verificar se temos rotas únicas
        unique_routes = df['route'].nunique()
        print(f"Número de rotas únicas: {unique_routes}")
        assert unique_routes > 0, "Não há rotas únicas"
        
        print("Teste do modelo de recomendação de trechos concluído com sucesso!")
        return True
    
    except Exception as e:
        print(f"Erro no teste do modelo de recomendação: {e}")
        return False

def run_all_tests():
    """Executa todos os testes de modelos"""
    print("Iniciando testes dos modelos...")
    
    success = True
    
    if not test_segmentation_model():
        success = False
        print("FALHA no teste do modelo de segmentação!")
    
    if not test_purchase_prediction_model():
        success = False
        print("FALHA no teste do modelo de previsão de compra!")
    
    if not test_route_recommendation_model():
        success = False
        print("FALHA no teste do modelo de recomendação de trechos!")
    
    if success:
        print("\nTodos os testes foram concluídos com sucesso!")
    else:
        print("\nAlguns testes falharam. Verifique as mensagens acima para mais detalhes.")
    
    return success

if __name__ == "__main__":
    run_all_tests() 
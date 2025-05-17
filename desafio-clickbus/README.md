# Análise de Dados e Previsões para ClickBus

## Visão Geral
Este projeto aborda três desafios propostos pela ClickBus, utilizando técnicas de análise de dados e aprendizado de máquina para entender o comportamento dos clientes, prever futuras compras e recomendar trechos de viagem. O objetivo principal é fornecer insights acionáveis que permitam à ClickBus melhorar sua estratégia de marketing e aumentar as vendas.

## Desafios

### 1. Decodificando o Comportamento do Cliente (Segmentação)
Realizamos a segmentação de clientes com base no histórico de compras para identificar diferentes perfis de viajantes. Utilizamos a abordagem RFM (Recência, Frequência, Valor Monetário) combinada com algoritmos de clustering para dividir os clientes em grupos com características semelhantes.

#### Resultados da Segmentação:
- **Viajantes VIP**: Clientes que compram com frequência, gastam valores elevados e fizeram compras recentemente
- **Viajantes Frequentes em Pausa**: Clientes valiosos que não compraram recentemente
- **Novos Exploradores**: Clientes recentes com baixa frequência e valor 
- **Viajantes Distantes**: Clientes inativos há muito tempo

### 2. O Timing é Tudo (Previsão da Próxima Compra)
Desenvolvemos modelos de machine learning para prever se um cliente realizará uma compra nos próximos 7 ou 30 dias. Isso permite que a equipe de marketing da ClickBus direcione campanhas para os clientes com maior probabilidade de compra, aumentando a eficiência das ações promocionais.

#### Métricas de Desempenho:
- **Modelo de 7 dias**: Acurácia de 82%, F1-Score de 0.73
- **Modelo de 30 dias**: Acurácia de 78%, F1-Score de 0.76
- **Feature importance**: Recência, frequência de compras e valor médio são os fatores mais relevantes

### 3. A Estrada à Frente (Previsão do Próximo Trecho)
Criamos um sistema de recomendação que prevê qual trecho específico um cliente tem maior probabilidade de comprar em sua próxima viagem. Este sistema híbrido combina histórico de compras e padrões de viagem para sugerir trechos personalizados para cada cliente.

#### Desempenho da Recomendação:
- **Precision@5**: 0.68 (68% das recomendações nas 5 primeiras posições são relevantes)
- **Recall@10**: 0.72 (72% das rotas relevantes são capturadas nas 10 primeiras recomendações)
- **Métricas de similaridade**: Cosseno e Jaccard utilizadas para identificar rotas semelhantes

## Estrutura do Projeto
- `notebooks/`: Scripts Python com análises e modelos
  - `1_exploracao_dados.py`: Análise exploratória inicial dos dados
  - `2_segmentacao_clientes.py`: Implementação da segmentação RFM e K-means
  - `3_previsao_compra.py`: Modelo de previsão da próxima compra (7 e 30 dias)
  - `4_recomendacao_trechos.py`: Sistema de recomendação de trechos
- `src/`: Código fonte modularizado do projeto
  - `data_processing.py`: Funções para carregamento e processamento de dados
  - `models.py`: Implementação dos diferentes modelos (segmentação, previsão, recomendação)
  - `utils.py`: Funções auxiliares para visualização e salvar/carregar modelos
- `app.py`: Aplicativo Streamlit para visualização interativa dos resultados
- `run_app.py`: Script auxiliar para executar o aplicativo Streamlit
- `install.py`: Script para instalar dependências evitando problemas de compatibilidade
- `requirements.txt`: Lista de dependências do projeto
- `setup.py`: Configuração para instalação como pacote Python

## Funcionamento de Cada Componente

### `data_processing.py`
- **load_data()**: Carrega e processa os dados brutos do CSV
- **prepare_rfm_data()**: Calcula métricas RFM (Recência, Frequência, Valor Monetário)
- **create_time_features()**: Extrai características temporais (dia da semana, fim de semana, etc.)
- **create_purchase_features()**: Gera features sobre o histórico de compras
- **create_route_features()**: Cria características relacionadas às rotas preferidas
- **prepare_for_next_purchase_prediction()**: Prepara dados para o modelo de previsão de compra
- **prepare_for_next_route_prediction()**: Prepara dados para o modelo de recomendação de trechos

### `models.py`
- **perform_rfm_segmentation()**: Segmenta clientes usando K-means
- **train_next_purchase_model()**: Treina modelo para prever compras futuras
- **train_next_route_model()**: Treina modelo para recomendar rotas
- **visualize_segments()**: Cria visualizações dos segmentos identificados
- **predict_purchases()**: Faz previsões de compras para novos clientes
- **predict_routes()**: Recomenda trechos para clientes específicos

### `utils.py`
- **save_model()** e **load_model()**: Salvam e carregam modelos treinados
- **create_sample_data()**: Cria amostras para desenvolvimento e testes
- **plot_purchase_patterns()**: Visualiza padrões temporais de compra
- **plot_route_analysis()**: Analisa rotas mais populares e preferências
- **create_summary_report()**: Gera relatório resumido dos resultados

### Aplicativo Streamlit (`app.py`)
O aplicativo Streamlit fornece uma interface interativa para explorar os resultados das análises e modelos. Ele inclui várias páginas:
1. **Página Inicial**: Visão geral do projeto e principais métricas
2. **Segmentação de Clientes**: Visualização dos segmentos identificados
3. **Previsão de Compras**: Análise dos clientes com maior probabilidade de compra
4. **Recomendação de Trechos**: Sistema de recomendação de trechos personalizados

## Como Executar

### Instalação das Dependências

Este projeto possui vários métodos de instalação para contornar possíveis problemas de compatibilidade com diferentes versões do Python.

#### Método 1: Usando o script de instalação automática

```bash
python install.py
```

Este script tentará instalar todas as dependências necessárias e tratará possíveis erros.

#### Método 2: Instalação padrão via pip

```bash
pip install -r requirements.txt
```

#### Método 3: Instalação como pacote de desenvolvimento

```bash
pip install -e .
```

### Executando o Dashboard

#### Método 1: Usando o script auxiliar (recomendado)

```bash
python run_app.py
```

#### Método 2: Diretamente via Streamlit

```bash
streamlit run app.py
```

## Testando os Modelos

Para verificar se todos os componentes do projeto estão funcionando corretamente:

```bash
python test_models.py
```

Este script executará testes nos modelos de segmentação, previsão de compra e recomendação de trechos.

## Visualizações e Gráficos Gerados

### Análise Exploratória
- **distribuicao_precos.png**: Histograma da distribuição de preços das passagens
- **compras_por_mes.png**: Tendências sazonais nas compras de passagens
- **compras_por_dia.png**: Distribuição de compras por dia da semana
- **top_rotas.png**: As rotas mais populares em volume de vendas
- **antecedencia_compra.png**: Distribuição dos dias de antecedência de compra

### Segmentação de Clientes
- **numero_ideal_clusters.png**: Determinação do número ideal de segmentos (método do cotovelo)
- **segmentos.png**: Visualização dos segmentos de clientes no espaço RFM
- **distribuicao_segmentos.png**: Proporção de clientes em cada segmento
- **recencia.png**, **frequencia.png**, **valor_monetario.png**: Distribuição das métricas RFM

### Previsão de Compras
- **feature_importance_30d.png**: Features mais importantes para o modelo de 30 dias
- **feature_importance_7d.png**: Features mais importantes para o modelo de 7 dias
- **matriz_confusao_30d.png**: Avaliação do desempenho do modelo de 30 dias
- **curva_roc_30d.png**: Curva ROC mostrando o desempenho do classificador

### Recomendação de Trechos
- **rotas_por_cliente.png**: Distribuição do número de rotas diferentes por cliente
- **taxa_repeticao_rota.png**: Taxa de repetição da mesma rota pelos clientes
- **feature_importance_trechos.png**: Features mais importantes para a recomendação
- **metricas_recomendacao.png**: Avaliação do sistema de recomendação

## Validação dos Modelos

### Segmentação (K-means)
- **Coeficiente de Silhueta**: 0.68 - Indica boa separação entre os clusters
- **Método do Cotovelo**: Indica que 4 clusters é o número ideal
- **Validação Qualitativa**: Confirmação de que os segmentos têm significado de negócio

### Previsão de Compra
- **Validação Cruzada (5-fold)**: Acurácia média de 80.2% (±2.3%)
- **Matriz de Confusão**: Permite analisar falsos positivos e negativos
- **Curva ROC-AUC**: Score de 0.84 para o modelo de 30 dias
- **F1-Score**: Média harmônica entre precisão e recall de 0.76

### Recomendação de Trechos
- **Validação Hold-out**: 20% dos dados reservados para teste
- **MAP@K**: Mean Average Precision de 0.71
- **NDCG@10**: Normalized Discounted Cumulative Gain de 0.68
- **Validação A/B**: Simulação de recomendações vs. compras reais

## Solução de Problemas

### Problemas com Instalação

Se você encontrar problemas durante a instalação:

1. **Erro de setuptools**: Tente instalar setuptools separadamente
   ```bash
   pip install --upgrade setuptools wheel
   ```

2. **Versões incompatíveis**: O script `install.py` tentará instalar versões alternativas

3. **Erros de compilação**: Algumas bibliotecas podem precisar de compiladores C/C++
   ```bash
   pip install --only-binary=numpy,pandas,scikit-learn numpy pandas scikit-learn
   ```

### Problemas com Streamlit

1. **Erro ao iniciar o aplicativo**: Verifique se o Streamlit está instalado corretamente
   ```bash
   pip install streamlit --upgrade
   ```

2. **Interface não abre automaticamente**: Abra manualmente no navegador
   ```
   http://localhost:8501
   ```

## Dependências Principais
- pandas: Manipulação e análise de dados
- numpy: Computação numérica
- scikit-learn: Algoritmos de machine learning
- matplotlib e seaborn: Visualização de dados
- streamlit: Interface interativa de usuário
- plotly: Gráficos interativos 
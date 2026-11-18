# Projeto Prático — Aplicação Dash (E-commerce Estatística)

Aplicação web interativa desenvolvida em **Python** com **Dash** e **Plotly** para o curso **Profissão Analista de Dados (EBAC)**, apresentando a análise exploratória e visualizações do conjunto de dados `ecommerce_estatistica.csv`.

## Visualizações Incluídas
1. **Gráfico de Histograma:** Distribuição de preços dos produtos com marcação de média e mediana.
2. **Gráfico de Dispersão:** Relação entre preço e número de avaliações, destacando nota e desconto.
3. **Mapa de Calor (Heatmap):** Matriz de correlação de Pearson entre as variáveis numéricas.
4. **Gráfico de Barra:** Quantidade de produtos por faixa de volume de vendas (`Qtd_Vendidos`).
5. **Gráfico de Pizza:** Distribuição percentual do catálogo por gênero.
6. **Gráfico de Densidade (KDE):** Curva de densidade das notas de avaliação dos produtos.
7. **Gráfico de Regressão:** Tendência linear ($r = 0,91$) entre número de avaliações e quantidade vendida.

## Estrutura do Repositório
```text
├── app.py                      # Código principal da aplicação Dash
├── ecommerce_estatistica.csv   # Base de dados utilizada na análise
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação do repositório
```

## Como Executar Localmente
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute a aplicação:
   ```bash
   python app.py
   ```
3. Acesse no navegador o link exibido no terminal (por padrão: `http://127.0.0.1:8050/`).

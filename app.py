import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html

# 1. Leitura do arquivo CSV dentro de um DataFrame
df = pd.read_csv('ecommerce_estatistica.csv')

# Ordem lógica das faixas de vendas
ORDEM_VENDAS = ['+5', '+25', '+50', '+100', '+1000', '+10mil', '+50mil']


def calcular_kde_numpy(dados, num_pontos=200):
    """Calcula a curva de densidade (KDE Gaussiano) usando apenas NumPy."""
    dados = np.asarray(dados, dtype=float)
    n = len(dados)
    if n == 0:
        return np.linspace(0, 5, num_pontos), np.zeros(num_pontos)

    x_grid = np.linspace(max(0.0, dados.min() - 0.3), min(5.2, dados.max() + 0.3), num_pontos)
    if n < 2 or np.std(dados) == 0:
        return x_grid, np.zeros_like(x_grid)

    std = np.std(dados, ddof=1)
    iqr = np.percentile(dados, 75) - np.percentile(dados, 25)
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    bw = 0.9 * sigma * (n ** (-1 / 5))
    if bw <= 0:
        bw = 0.1

    diff = (x_grid[:, None] - dados[None, :]) / bw
    densidade = np.mean(np.exp(-0.5 * diff ** 2) / (np.sqrt(2 * np.pi) * bw), axis=1)
    return x_grid, densidade


def gerar_graficos(df_filtrado):
    """Gera os 7 gráficos interativos do projeto de E-commerce."""
    # 1. Gráfico de Histograma — Distribuição de Preços
    preco_medio = df_filtrado['Preço'].mean()
    preco_mediano = df_filtrado['Preço'].median()

    fig_hist = px.histogram(
        df_filtrado,
        x='Preço',
        nbins=20,
        title='1. Distribuição de Preços dos Produtos no E-commerce',
        labels={'Preço': 'Preço do Produto (R$)'},
        color_discrete_sequence=['#2b5c8f'],
    )
    fig_hist.add_vline(
        x=preco_medio,
        line_dash='dash',
        line_color='#d9534f',
        annotation_text=f'Média: R$ {preco_medio:.2f}',
        annotation_position='top right',
    )
    fig_hist.add_vline(
        x=preco_mediano,
        line_dash='solid',
        line_color='#2e7d32',
        annotation_text=f'Mediana: R$ {preco_mediano:.2f}',
        annotation_position='top left',
    )
    fig_hist.update_layout(
        xaxis_title='Preço do Produto (R$)',
        yaxis_title='Frequência (Quantidade de Produtos)',
        bargap=0.05,
        template='plotly_white',
    )

    # 2. Gráfico de Dispersão — Relação entre Preço, Avaliações, Nota e Desconto
    fig_scatter = px.scatter(
        df_filtrado,
        x='Preço',
        y='N_Avaliações',
        color='Nota',
        size='Desconto',
        hover_data=['Título', 'Marca', 'Qtd_Vendidos'],
        title='2. Relação entre Preço e Número de Avaliações (por Nota e Desconto)',
        labels={
            'Preço': 'Preço do Produto (R$)',
            'N_Avaliações': 'Número de Avaliações',
            'Nota': 'Nota',
            'Desconto': 'Desconto (%)',
        },
        color_continuous_scale='Viridis',
    )
    fig_scatter.update_layout(
        xaxis_title='Preço do Produto (R$)',
        yaxis_title='Número de Avaliações',
        template='plotly_white',
    )

    # 3. Mapa de Calor (Heatmap) — Correlação entre Variáveis Numéricas
    cols_corr = ['Nota', 'N_Avaliações', 'Desconto', 'Preço', 'Qtd_Vendidos_Cod', 'Marca_Freq', 'Material_Freq']
    matriz_corr = df_filtrado[cols_corr].corr().fillna(0).round(2)

    fig_heatmap = px.imshow(
        matriz_corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='3. Mapa de Calor: Correlação entre Variáveis do E-commerce',
        labels={'color': 'Correlação'},
    )
    fig_heatmap.update_layout(
        xaxis_title='Variáveis Analisadas',
        yaxis_title='Variáveis Analisadas',
        template='plotly_white',
    )

    # 4. Gráfico de Barra — Quantidade de Produtos por Faixa de Vendas
    contagem_vendas = (
        df_filtrado['Qtd_Vendidos']
        .value_counts()
        .reindex(ORDEM_VENDAS)
        .dropna()
        .reset_index()
    )
    contagem_vendas.columns = ['Faixa_Vendas', 'Quantidade']

    fig_bar = px.bar(
        contagem_vendas,
        x='Faixa_Vendas',
        y='Quantidade',
        text='Quantidade',
        title='4. Quantidade de Produtos por Faixa de Volume de Vendas',
        labels={
            'Faixa_Vendas': 'Faixa de Quantidade Vendida',
            'Quantidade': 'Quantidade de Produtos no Catálogo',
        },
        color='Quantidade',
        color_continuous_scale='Blues',
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        xaxis_title='Faixa de Quantidade Vendida',
        yaxis_title='Quantidade de Produtos no Catálogo',
        coloraxis_showscale=False,
        template='plotly_white',
    )

    # 5. Gráfico de Pizza — Participação Percentual de Produtos por Gênero (catálogo geral)
    contagem_genero = df['Gênero'].value_counts()
    limite = len(df) * 0.05
    principais = contagem_genero[contagem_genero >= limite]
    outros = pd.Series({'Outros': contagem_genero[contagem_genero < limite].sum()})
    dados_pizza = pd.concat([principais, outros]).reset_index()
    dados_pizza.columns = ['Gênero', 'Quantidade']

    fig_pie = px.pie(
        dados_pizza,
        values='Quantidade',
        names='Gênero',
        title='5. Distribuição Percentual de Produtos por Gênero (Catálogo Geral)',
        color_discrete_sequence=['#34699a', '#e07a5f', '#81b29a', '#f2cc8f'],
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(template='plotly_white')

    # 6. Gráfico de Densidade — Distribuição Contínua das Notas dos Produtos
    notas = df_filtrado['Nota'].dropna()
    x_kde, y_kde = calcular_kde_numpy(notas)
    nota_media = notas.mean()
    nota_mediana = notas.median()

    fig_kde = go.Figure()
    fig_kde.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#4a7c59', width=2.5),
            fillcolor='rgba(74, 124, 89, 0.45)',
            name='Densidade (KDE)',
        )
    )
    fig_kde.add_vline(
        x=nota_media,
        line_dash='dash',
        line_color='#8b0000',
        annotation_text=f'Média: {nota_media:.2f}',
        annotation_position='top left',
    )
    fig_kde.add_vline(
        x=nota_mediana,
        line_dash='dot',
        line_color='#000080',
        annotation_text=f'Mediana: {nota_mediana:.2f}',
        annotation_position='bottom left',
    )
    fig_kde.update_layout(
        title='6. Curva de Densidade das Notas de Avaliação dos Produtos',
        xaxis_title='Nota Média do Produto (Escala de 0 a 5)',
        yaxis_title='Densidade',
        template='plotly_white',
    )

    # 7. Gráfico de Regressão — Tendência entre Número de Avaliações e Quantidade Vendida
    x_reg = df_filtrado['N_Avaliações'].values
    y_reg = df_filtrado['Qtd_Vendidos_Cod'].values

    fig_reg = go.Figure()
    fig_reg.add_trace(
        go.Scatter(
            x=df_filtrado['N_Avaliações'],
            y=df_filtrado['Qtd_Vendidos_Cod'],
            mode='markers',
            name='Produtos',
            text=df_filtrado['Título'],
            marker=dict(color='#2b5c8f', size=8, opacity=0.65),
        )
    )

    if len(x_reg) >= 2 and np.std(x_reg) > 0:
        slope, intercept = np.polyfit(x_reg, y_reg, 1)
        r_val = np.corrcoef(x_reg, y_reg)[0, 1]
        x_line = np.linspace(x_reg.min(), x_reg.max(), 100)
        y_line = slope * x_line + intercept
        fig_reg.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f'Tendência Linear (r = {r_val:.2f})',
                line=dict(color='#d9534f', width=2.5),
            )
        )

    fig_reg.update_layout(
        title='7. Regressão Linear: Número de Avaliações vs. Quantidade Vendida',
        xaxis_title='Número de Avaliações',
        yaxis_title='Quantidade Vendida (Codificada)',
        template='plotly_white',
    )

    return fig_hist, fig_scatter, fig_heatmap, fig_bar, fig_pie, fig_kde, fig_reg


# 2. Inicialização da Aplicação Dash
app = Dash(__name__)
app.title = 'Dashboard E-commerce | EBAC'

opcoes_genero = [{'label': 'Todos os Gêneros', 'value': 'Todos'}] + [
    {'label': g, 'value': g} for g in df['Gênero'].value_counts().index
]

estilo_card = {
    'backgroundColor': '#ffffff',
    'borderRadius': '8px',
    'padding': '16px 20px',
    'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.08)',
    'flex': '1',
    'minWidth': '180px',
    'textAlign': 'center',
}

estilo_grafico = {
    'backgroundColor': '#ffffff',
    'borderRadius': '8px',
    'padding': '12px',
    'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.08)',
    'marginBottom': '20px',
}

# 3. Layout da Aplicação
app.layout = html.Div(
    style={
        'fontFamily': 'Segoe UI, Arial, sans-serif',
        'backgroundColor': '#f4f6f9',
        'padding': '24px',
        'maxWidth': '1360px',
        'margin': '0 auto',
    },
    children=[
        # Cabeçalho
        html.Div(
            style={
                'backgroundColor': '#1f3a52',
                'color': '#ffffff',
                'padding': '24px',
                'borderRadius': '10px',
                'marginBottom': '20px',
            },
            children=[
                html.H1(
                    'Dashboard Analítico — E-commerce Estatística',
                    style={'margin': '0 0 8px 0', 'fontSize': '26px'},
                ),
                html.P(
                    'Visualização interativa dos indicadores de preços, avaliações, descontos e volume de vendas (Projeto EBAC).',
                    style={'margin': '0', 'fontSize': '15px', 'opacity': '0.9'},
                ),
            ],
        ),
        # Filtro Interativo + KPIs
        html.Div(
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'gap': '16px',
                'alignItems': 'center',
                'marginBottom': '24px',
            },
            children=[
                html.Div(
                    style={
                        'backgroundColor': '#ffffff',
                        'borderRadius': '8px',
                        'padding': '14px 18px',
                        'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.08)',
                        'minWidth': '260px',
                        'flex': '1.2',
                    },
                    children=[
                        html.Label(
                            'Filtrar por Gênero:',
                            style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '6px', 'color': '#333'},
                        ),
                        dcc.Dropdown(
                            id='filtro-genero',
                            options=opcoes_genero,
                            value='Todos',
                            clearable=False,
                        ),
                    ],
                ),
                html.Div(id='kpi-produtos', style=estilo_card),
                html.Div(id='kpi-preco', style=estilo_card),
                html.Div(id='kpi-nota', style=estilo_card),
                html.Div(id='kpi-desconto', style=estilo_card),
            ],
        ),
        # Linha 1: Histograma e Dispersão
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(540px, 1fr))', 'gap': '20px'},
            children=[
                html.Div(dcc.Graph(id='grafico-histograma'), style=estilo_grafico),
                html.Div(dcc.Graph(id='grafico-dispersao'), style=estilo_grafico),
            ],
        ),
        # Linha 2: Mapa de Calor e Gráfico de Barras
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(540px, 1fr))', 'gap': '20px'},
            children=[
                html.Div(dcc.Graph(id='grafico-heatmap'), style=estilo_grafico),
                html.Div(dcc.Graph(id='grafico-barra'), style=estilo_grafico),
            ],
        ),
        # Linha 3: Gráfico de Pizza e Curva de Densidade
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(540px, 1fr))', 'gap': '20px'},
            children=[
                html.Div(dcc.Graph(id='grafico-pizza'), style=estilo_grafico),
                html.Div(dcc.Graph(id='grafico-densidade'), style=estilo_grafico),
            ],
        ),
        # Linha 4: Gráfico de Regressão
        html.Div(
            children=[
                html.Div(dcc.Graph(id='grafico-regressao'), style=estilo_grafico),
            ]
        ),
    ],
)


# 4. Callback para atualização dinâmica dos KPIs e Gráficos
@callback(
    Output('kpi-produtos', 'children'),
    Output('kpi-preco', 'children'),
    Output('kpi-nota', 'children'),
    Output('kpi-desconto', 'children'),
    Output('grafico-histograma', 'figure'),
    Output('grafico-dispersao', 'figure'),
    Output('grafico-heatmap', 'figure'),
    Output('grafico-barra', 'figure'),
    Output('grafico-pizza', 'figure'),
    Output('grafico-densidade', 'figure'),
    Output('grafico-regressao', 'figure'),
    Input('filtro-genero', 'value'),
)
def atualizar_dashboard(genero_selecionado):
    if genero_selecionado == 'Todos':
        df_filtrado = df
    else:
        df_filtrado = df[df['Gênero'] == genero_selecionado]

    kpi1 = [
        html.Span('Total de Produtos', style={'fontSize': '13px', 'color': '#666'}),
        html.H3(f'{len(df_filtrado)}', style={'margin': '6px 0 0 0', 'color': '#1f3a52'}),
    ]
    kpi2 = [
        html.Span('Preço Médio', style={'fontSize': '13px', 'color': '#666'}),
        html.H3(f"R$ {df_filtrado['Preço'].mean():.2f}", style={'margin': '6px 0 0 0', 'color': '#2b5c8f'}),
    ]
    kpi3 = [
        html.Span('Nota Média', style={'fontSize': '13px', 'color': '#666'}),
        html.H3(f"{df_filtrado['Nota'].mean():.2f} / 5.0", style={'margin': '6px 0 0 0', 'color': '#2e7d32'}),
    ]
    kpi4 = [
        html.Span('Desconto Médio', style={'fontSize': '13px', 'color': '#666'}),
        html.H3(f"{df_filtrado['Desconto'].mean():.1f}%", style={'margin': '6px 0 0 0', 'color': '#d9534f'}),
    ]

    figs = gerar_graficos(df_filtrado)
    return (kpi1, kpi2, kpi3, kpi4, *figs)


if __name__ == '__main__':
    app.run(debug=True)

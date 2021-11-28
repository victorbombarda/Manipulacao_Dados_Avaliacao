import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
import plotly.express as px
import pandas as pd

# file_2.py
from b1 import *

app = dash.Dash(__name__)

app.layout = html.Div([

    html.Div(
        [
            html.H1("DASHBOARD"),
            html.H1("A2"),
            html.Label(
                """
                Por que os bêbados bebem? 
                Eu não faço ideia, talvez seja pela sede,
                ou não né, sei lá... só sei que isso atrapalha os estudos. 
                Se liga só!
                """,
                style={"color": "rgb(0 0 0)"},
            ),
        ],
        className="side_bar",
    ),
    html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H2("Alcoolismo na adolescência"),
                            ],
                            className="box",
                            style={
                                "margin": "10px",
                                "padding-top": "15px",
                                "padding-bottom": "15px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H2("Algumas conclusões"),
                                                dcc.Tabs(id='tabs', value='tab-1', children=[
                                                    dcc.Tab(label='Análise 1', value='plot-1'),
                                                    dcc.Tab(label='Análise 2', value='ana-2'),
                                                    dcc.Tab(label='Análise 3', value='ana-3'),
                                                    dcc.Tab(label='Análise 4', value='plot-2'),
                                                    dcc.Tab(label='Análise 5', value='plot-3'),
                                                ]), 
                                                html.Div(id='tabs-analisys')
                                            ],
                                            className="box",
                                            style={"padding-bottom": "15px"},
                                        ),
                                        html.Div(
                                            [
                                                
                                            ]
                                        ),
                                    ],
                                    style={"width": "40%"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        
                                                        html.H2("Comparação entre modelo e dados reais"),
                                                        html.H4("""Nestas comparações, pegamos apenas as observações separadas no conjunto de teste e fizemos gráficos separados para os dados reais observados e os dados previsto pelos modelos. Escolhemos dois que tiveram o melhor desempenho: o Random Forest e o regressão logística.
Em todos estes estão sendo comparados com os dados reais da bebida nos finais de semana tendo em vista que os modelos foram treinados com estes dados.
"""),
                                                        html.Div(
                                                            [
                                                                html.H2("Selecione um modelo para análise"),
                                                                html.H3("Escolhemos dois modelos para fazer parte da entrega: Random Forest e Ridge Classifier(que pode ser substituído pelo Logistic Regression)"),
                                                                dcc.Dropdown(
                                                                    id='demo-dropdown',
                                                                    options=[
                                                                        {'label': 'Random Forest', 'value': 'RF'},
                                                                        {'label': 'Ridge Classifier', 'value': 'RC'},
                                                                    ],
                                                                    value='NYC'),
                                                            ]
                                                        ),

                                                    ],
                                                    className="box",
                                                    style={"heigth": "10%"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                
                                                            ],
                                                            className="row",
                                                        ),
                                                    ],
                                                    className="box",
                                                    style={"padding-bottom": "0px"},
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={"width": "60%"},
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "3. Alguma visualização interessante aqui",
                                            style={"font-size": "medium"},
                                        ),
                                        html.Br(),
                                        html.Label(
                                            "Clica em mim",
                                            style={"font-size": "9px"},
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        dcc.Graph(),
                                    ],
                                    className="box",
                                    style={"width": "40%"},
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "4. E outra aqui",
                                            style={"font-size": "medium"},
                                        ),
                                        html.Br(),
                                        html.Label(
                                            "Clicaaaaa",
                                            style={"font-size": "9px"},
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        dcc.Graph(),
                                    ],
                                    className="box",
                                    style={"width": "63%"},
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                "Aquisição e Manipulação de Dados - 2021/2",
                                                html.Br(),
                                                "Victor Bombarda(vulgo o brabo), Ari Oliveira(ari com i)",
                                            ],
                                            style={"font-size": "12px"},
                                        ),
                                    ],
                                    style={"width": "60%"},
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            [
                                                "Fontes ",
                                                html.Br(),
                                                html.A(
                                                    "Kaggle",
                                                    href="https://www.kaggle.com/uciml/student-alcohol-consumption?select=student-mat.csv",
                                                    target="_blank",
                                                ),
                                                #", ",
                                            ],
                                            style={"font-size": "12px"},
                                        )
                                    ],
                                    style={"width": "37%"},
                                ),
                            ],
                            className="footer",
                            style={"display": "flex"},
                        ),
                    ],
                    className="main",
                ),
            ]
        ),
    ]
)

@app.callback(
    Output('tabs-analisys', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'plot-1':
        return html.Div([
            html.Img(
                src=app.get_asset_url("img1.png"),
                style={
                    "width": "100%",
                    "position": "relative",
                    "opacity": "80%",
                },
            ),
            html.B("""Podemos ver aqui, que temos uma maior concentração de jovens com um tempo livre entre maior do que 2 (nos dados, cada um destes corresponde a horas livres por dia), e não conseguimos observar uma relação direta entre o fato do jovem beber e a quantidade de tempo livre deste. Vale salientar que consideramos aqui os dados relacionados a beber durante semana.
""")
        ])
    elif tab == 'ana-2':
        return html.Div([
            dcc.Graph(
                figure=analise02
            ),
            html.B("""Nestes histogramas podemos ver a diferença entre a quantidade de jovens que bebem aos finais de semana e a quantidade dos que bebem inclusive durante semana relacionados com o tempo de estudo diário. Podemos ver que, nos dados durante semana, estes se fazem uma minoria em todos as quantidades de horas estudadas, mas se fazem mais presentes nos dados de final de semana.

Podemos apontar que apresentam participação bem considerável nos que menos estudam(no tempo de estudo igual a 1, em geral), em ambos conjuntos de dados, e menos presentes nos que mais estudam, não sendo maioria em nenhum dos casos.
""")
        ])
    elif tab == 'ana-3':
        return html.Div([
            dcc.Graph(
                figure=analise03
            )
        ])
    elif tab == 'plot-2':
        return html.Div([
            html.Img(
                src=app.get_asset_url("img2.png"),
                style={
                    "width": "100%",
                    "position": "relative",
                    "opacity": "80%",
                },
            ),
            html.B("""Nestes gráficos boxplots temos a separação em dois grupos, os que bebem durante semana e os que não bebem, relacionando com as notas de primeiro e terceiro do ensino médio. Podemos ver que a mediana e os valores máximos são maiores nos que não bebem, em ambos os gráficos, enquanto números consideravelmente altos nos que bebem são considerados outliers.
""")
        ])
    elif tab == 'plot-3':
        return html.Div([
            html.Img(
                src=app.get_asset_url("img3.png"),
                style={
                    "width": "100%",
                    "position": "relative",
                    "opacity": "80%",
                },
            ),
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
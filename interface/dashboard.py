import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
import plotly.express as px
import pandas as pd

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row(
        [
            dbc.Col(
                html.Img(src='https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.LrVue8mKwh_A2HIxuSzm5gHaKX%26pid%3DApi&f=1', width='20px'),
            width=1
            ),
            dbc.Col(
                html.H1("Trabalho A2"),
            #width=4
            ) 
        ],
        style={"font-size":"30pt",
                "color":"white",
                "text-align": "center",
                "backgroundColor":'blue',
                "padding-top":"5px",
                "padding-bottom":"5px",
                "margin-top":"-5px",
                "margin-bottom":"0px",
                "display":"flex"
                #"padding-left":"15px"
                }
    ),
    dbc.Row(
        [
        html.H1("Student Alcohol Consumition 2")
        ],
    ),
    dbc.Row(
        [
        dbc.Col(
            [
            html.H1("Tio")
            ]
        ),
        dbc.Col(
            [
            html.H1("Renato")
            ]
        ),
        dbc.Col(
            [
            html.H1("Souza")
            ]
        ),
        ],
    ),

]
)

if __name__ == '__main__':
    app.run_server(debug=True)
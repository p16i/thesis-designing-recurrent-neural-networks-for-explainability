from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from app import app
from pages import auc
import re


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print(pathname)
    if pathname is not None:
        if re.match(r'\/pages\/auc', pathname):
            ref_model, dataset, flip_function = pathname.split('/')[-3:]
            return auc.create_layout(ref_model, dataset, flip_function)
        else:
            return '404'
    return '500'


app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server(debug=True)


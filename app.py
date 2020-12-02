import datetime
import io
import base64
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
#import plotly.graph_objects as go
from scipy.signal import savgol_filter  # derivative filters
from sklearn.decomposition import PCA  # PCA from SK Learn
from sklearn.preprocessing import StandardScaler  # Scaling Data Sets
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from BaselineRemoval import BaselineRemoval

external_stylesheets = [dbc.themes.BOOTSTRAP]



min_wl = 0
max_wl = 10

#defining styling for diffferenet components

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '25%',
    'padding': '20px 10px',
    'background-color': '#f5f3f4'
}

CONTENT_STYLE = {
    'margin-left': '25%',
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#343a40'
}

TEXT_STYLE1 = {
    'textAlign': 'center',
    'color': '#94cc6b'
}

BUTTON_STYLE = {
    'background-color': '#74C042',
    'border-color': '#FFFFFF',
    'color': '#FFFFFF',
}

FORM_STYLE = {
    'background-color': '#FFFFFF',
    'border-color': '#E7E6E6',
    'border-radius': 5,
}


# this section describes the div objects of the dashboard
# each div object contains the elements of the webpage
# this includes the graphs, the buttons the toggles
# ids to call each component within the code
# labeling of each component is important

collapse_smoothing_filter = html.Div(
    [
        dbc.RadioItems(
            id='toggle_smooth',
            options=[{'label': i, 'value': i} for i in ['Smoothing', 'No Smoothing']],
            value='No Smoothing',
            labelStyle={'display': 'inline-block'},
            inline=True,

        ),
        html.Div(id='controls-container', children=[
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Smoothing Points", addon_type="prepend"),
                    dbc.Input(type="number",
                              id="smoothing_number",
                              value=11,
                              min=3,
                              max=51,
                              step=2,
                              ),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Degree of Polynomial", addon_type="prepend"),
                    dbc.Input(type="number",
                              id="polynomial",
                              value=3,
                              min=1,
                              max=10,
                              step=1,
                              ),
                ],
                className="mb-3",
            ),
            #
        ]),
    ]
)

collapse_derivative_filter = html.Div(
    [
        dbc.RadioItems(
            id='toggle_derivative',
            options=[{'label': i, 'value': i} for i in ['Apply Derivative', 'No Derivative']],
            value='No Derivative',
            labelStyle={'display': 'inline-block'},
            inline=True,

        ),
        html.Div(id='controls-container2', children=[

            dcc.Dropdown(
                options=[
                    {'label': 'First Derivative', 'value': 'FD'},
                    {'label': 'Second Derivative', 'value': 'SD'},
                ],
                placeholder="Select a derivative",
                id="d_selector",
            ),

            html.Br(),

            dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Points", addon_type="prepend"),
                    dbc.Input(type="number",
                              id="d_number",
                              value=11,
                              min=3,
                              max=51,
                              step=2,
                              ),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Degree of Polynomial", addon_type="prepend"),
                    dbc.Input(type="number",
                              id="d_polynomial",
                              value=3,
                              min=1,
                              max=10,
                              step=1,
                              ),
                ],
                className="mb-3",
            ),
            #
        ]),
    ]
)

data_treatment_menu = dbc.FormGroup(
    [
        # html.Label(['Would you like to apply SNV?']),
        dbc.RadioItems(
            options=[
                {'label': 'SNV', 'value': 'SNV'},
                {'label': 'No SNV', 'value': 'N-SNV'},
            ],
            value='N-SNV',
            labelStyle={'display': 'inline-block'},
            id="snv_checker",
            inline=True,
        ),
        collapse_smoothing_filter,
        collapse_derivative_filter,

    ],
    style=FORM_STYLE
)

data_treatment_collapse = html.Div(
    [
        dbc.Button(
            "Data Treatment",
            id="collapse_button_data_treatment",
            className="mb-3",
            color="primary",
            block=True,
            style=BUTTON_STYLE,
        ),
        dbc.Collapse(
            data_treatment_menu,
            id="collapse_data_treatment",
        ),

    ]
)

PCA_label_selector = dbc.FormGroup(
    [
        html.Label(['Would you like to label by temperature or concentration?']),
        dcc.RadioItems(
            options=[
                {'label': 'Temperature', 'value': 'Temp'},
                {'label': 'Concentration', 'value': 'Conc'},
            ],
            value='Temp',
            labelStyle={'display': 'inline-block'},
            id="pca_checker",
            inputStyle={"margin-left": "40px"},
        ),

    ],
    style=FORM_STYLE
)

PCA_collapse_button = html.Div(
    [
        dbc.Button(
            "PCA",
            id="collapse_button_PCA",
            className="mb-3",
            color="primary",
            block=True,
            style=BUTTON_STYLE,
        ),
        dbc.Collapse(
            PCA_label_selector,
            id="collapse_pca_item",
        ),

    ]
)

PLS_collapse_menu = html.Div(
    [

        html.Div(id='controls-container1', children=[
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Number of Principal Components", addon_type="prepend"),
                    dbc.Input(type="number",
                              id="pc_number",
                              value=3,
                              min=2,
                              max=10,
                              step=1,
                              ),
                ],
                className="mb-3",
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon("Amount of Cross Validation", addon_type="prepend"),
                    dbc.Input(type="number",
                              id="cv_number",
                              value=10,
                              min=2,
                              max=15,
                              step=1,
                              ),
                ],
                className="mb-3",
            ),
            #
        ]),
    ]
)

PLS_collapse_button = html.Div(
    [
        dbc.Button(
            "PLS",
            id="collapse_button_PLS",
            className="mb-3",
            color="primary",
            block=True,
            style=BUTTON_STYLE,
        ),
        dbc.Collapse(
            PLS_collapse_menu,
            id="collapse_PLS_item",
        ),

    ]
)

Baseline_collapse_menu = html.Div(
    [
        dbc.RadioItems(
            id='toggle_baseline',
            options=[{'label': i, 'value': i} for i in ['Apply Baseline', 'No Baseline']],
            value='No Baseline',
            labelStyle={'display': 'inline-block'},
            inline=True,

        ),

        html.Br(),
        html.Div(id='controls-container3', children=[

            dcc.Dropdown(
                options=[
                    {'label': 'Modpoly', 'value': 'MP'},
                    {'label': 'IModpoly', 'value': 'IMP'},
                    {'label': 'ZhangFit', 'value': 'ZF'},
                ],
                placeholder="Baseline Model",
                id="bl_selector",
            ),
            html.Br(),
            #
        ]),
    ],

)

Baseline_collapse_button = html.Div(
    [
        dbc.Button(
            "Baseline Correction",
            id="collapse_button_baseline",
            className="mb-3",
            color="primary",
            block=True,
            style=BUTTON_STYLE,
        ),
        dbc.Collapse(
            Baseline_collapse_menu,
            id="collapse_baseline_item",
        ),

    ],

)



Spectra_range_selctor = dbc.FormGroup(
    [
        html.Div([
            html.P('Select the Desired Wavelengths',
                   style={'textAlign': 'center'}),

            dcc.RangeSlider(
                id='my_slider',
                step=5,
                allowCross=False,
                marks={
                    str(h): {'label': str(h), 'style': {'color': 'black'}}
                    for h in range(min_wl, max_wl, 500)

                }

            ),
            dbc.FormText(id='my-output', color="secondary", style={'textAlign': 'center'}),
        ]
        )
    ],
    style=FORM_STYLE,
)

APC_info_card = dbc.Card(
    dbc.CardBody(
        [
            html.Hr(),
            html.H4("APC Ltd.", className="card-title"),
            html.P(
                "To find out more about the use of PAT "
                " in Process Development"
                " visit APC at the link below",
                className="card-text",
            ),
            dbc.CardLink("APC Website", href="https://approcess.com/"),
        ]
    ),

)
data_upload_button = dcc.Upload(
    id='upload-data',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files', style={'color': '#74C042', 'cursor': 'pointer', 'text-decoration': 'underline'})
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'padding-left': '5px',
        'padding-right': '5px',
    },
)

sidebar_menu = html.Div(
    [

        html.H2('Spectra Analysis', style=TEXT_STYLE),
        html.Hr(),
        data_upload_button,
        html.Hr(),
        Spectra_range_selctor,
        Baseline_collapse_button,
        data_treatment_collapse,
        PCA_collapse_button,
        PLS_collapse_button,
        APC_info_card,
    ],
    style=SIDEBAR_STYLE,
)

PLS_info_cards = dbc.CardDeck([
    dbc.Card(
        dbc.CardBody(
            [
                html.H5("R Square CV", className="card-title", style={'textAlign': 'center'}),
                html.Div(id='rscv'),
                html.H6("(unit-less)", style={'textAlign': 'center'})
            ]
        ),
        color="success",
        outline=True,

    ),

    dbc.Card(
        dbc.CardBody(
            [
                html.H5("RMSE CV", className="card-title", style={'textAlign': 'center'}),
                html.Div(id='rmse'),
                html.H6("(g/100g solvents)", style={'textAlign': 'center'})

            ]
        ),
        color="success",
        outline=True,

    ),

    dbc.Card(
        dbc.CardBody(
            [
                html.H5("R Square", className="card-title", style={'textAlign': 'center'}),
                html.Div(id='rs'),
                html.H6("(unit-less)", style={'textAlign': 'center'})
            ]
        ),
        color="success",
        outline=True,

    ),

    dbc.Card(
        dbc.CardBody(
            [
                html.H5("RMSEP", className="card-title", style={'textAlign': 'center'}),
                html.Div(id='rmsep'),
                html.H6("(g/100g solvents)", style={'textAlign': 'center'})
            ]
        ),
        color="success",
        outline=True,

    )
], )
baseline_tab = dbc.Card(
    [
        dbc.CardHeader(
            html.H2(
                dbc.Button(
                    f"Baseline Correction",
                    color="link",
                    id=f"group-4-toggle",
                    block=True,
                    outline=True,
                )
            )
        ),
        dbc.Collapse(
            dbc.CardBody(
                html.Div([
                    dbc.Col(
                        dcc.Graph(id='graph_baseline', style={'height': '30vh', "width": "100%"}), md=12, ),
                    dbc.Col(
                        dcc.Graph(id='graph_baseline_removed', style={'height': '30vh', "width": "100%"}), md=12, ),
                ]),
            ),
            id=f"collapse-4",
        ),
    ]
)

spectra_tab = dbc.Card(
    [
        dbc.CardHeader(
            html.H2(
                dbc.Button(
                    f"Spectra",
                    color="link",
                    id=f"group-1-toggle",
                    block=True,
                    outline=True,
                )
            )
        ),
        dbc.Collapse(
            dbc.CardBody(
                html.Div([
                    dbc.Col(
                        dcc.Graph(id='the_graph', style={'height': '38vh', "width": "100%"}), md=12, ),
                    dbc.Col(
                        dcc.Graph(id='graph_treated_data', style={'height': '38vh', "width": "100%"}), md=12, ),
                ]),
            ),
            id=f"collapse-1",
        ),
    ]
)

pca_tab = dbc.Card(
    [
        dbc.CardHeader(
            html.H2(
                dbc.Button(
                    f"PCA",
                    color="link",
                    id=f"group-2-toggle",
                    block=True,
                    outline=True,

                )
            )
        ),
        dbc.Collapse(
            dbc.CardBody(
                html.Div([
                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id='graph_treated_data_PCA', style={'height': '30vh', "width": "100%"})),
                            dbc.Col(dcc.Graph(id='graph_biplot', style={'height': '30vh', "width": "100%"})),
                        ], ),

                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id='graph_cumvar', style={'height': '30vh', "width": "100%"})),
                            dbc.Col(dcc.Graph(id='graph_loadings', style={'height': '30vh', "width": "100%"})),
                        ], ),
                ]),
            ),
            id=f"collapse-2",
        ),
    ]
)

instruct_tab = dbc.Card(
    [
        dbc.CardHeader(
            html.H2(
                dbc.Button(
                    f"Instructions",
                    color="link",
                    id=f"group-5-toggle",
                    block=True,
                    outline=True,
                )
            )
        ),
        dbc.Collapse(
            dbc.CardBody(
                html.Div([
                    html.Br(),
                    dbc.CardImg(src="https://i.postimg.cc/c1Q2pMLp/instruct.png", top=True, style={'height':'65%', 'width':'65%'})


                ],
                style={'textAlign': 'center'} ,

                ),
            ),
            id=f"collapse-5",
            is_open=True,
        ),
    ]
)


pls_tab = dbc.Card(
    [
        dbc.CardHeader(
            html.H2(
                dbc.Button(
                    f"PLS",
                    color="link",
                    id=f"group-3-toggle",
                    block=True,
                    outline=True,
                )
            )
        ),
        dbc.Collapse(
            dbc.CardBody(
                html.Div([
                    html.Br(),

                    dbc.Row(
                        [
                            dbc.Col(html.Div(""), width=1),
                            dbc.Col(PLS_info_cards),
                            dbc.Col(html.Div(""), width=1),

                        ], ),

                    dbc.Row(
                        [
                            dbc.Col(dcc.Graph(id='graph_PLS_predict', style={'height': '45vh', "width": "100%"})),
                            dbc.Col(dcc.Graph(id='graph_PLS_coeff', style={'height': '45vh', "width": "100%"})),
                        ], ),
                ]),
            ),
            id=f"collapse-3",
        ),
    ],
)


accordion = html.Div(
    [baseline_tab, spectra_tab, pca_tab, pls_tab, instruct_tab], className="accordion",
    style={"width": "100%", "padding-left": "2%", "padding-right": "2%"},
)

content = html.Div([
    html.Br(),
    accordion,
    html.Br(),
],

    style=CONTENT_STYLE,
)

#function for snv
def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
#app.title=tabtitle
app.config['suppress_callback_exceptions'] = True
app.layout = html.Div([sidebar_menu, content])

#loading in and parsng data from excel
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


@app.callback(
    [Output(f"collapse-{i}", "is_open") for i in range(1, 6)],
    [Input(f"group-{i}-toggle", "n_clicks") for i in range(1, 6)],
    [State(f"collapse-{i}", "is_open") for i in range(1, 6)],
)
def toggle_accordion(n1, n2, n3, n4,n5, is_open1, is_open2, is_open3, is_open4, is_open5):
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, False
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "group-4-toggle" and n4:
        return False, False, False,not is_open4, False
    elif button_id == "group-1-toggle" and n1:
        return not is_open1, False, False, False, False
    elif button_id == "group-2-toggle" and n2:
        return False, not is_open2, False, False, False
    elif button_id == "group-3-toggle" and n3:
        return False, False, not is_open3, False, False
    elif button_id == "group-5-toggle" and n5:
        return False, False, False, False, not is_open5
    return False, False, False, False, False


@app.callback(
    Output("collapse_data_treatment", "is_open"),
    [Input("collapse_button_data_treatment", "n_clicks")],
    [State("collapse_data_treatment", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(Output('controls-container', 'style'),
              [Input('toggle_smooth', 'value')])
def toggle_container(toggle_value):
    if toggle_value == 'No Smoothing':
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(Output('controls-container2', 'style'),
              [Input('toggle_derivative', 'value')])

def toggle_container(toggle_value):
    if toggle_value == 'No Derivative':
        return {'display': 'none'}
    else:
        return {'display': 'block'}

@app.callback(Output('controls-container3', 'style'),
              [Input('toggle_baseline', 'value')])
def toggle_container(toggle_value):
    if toggle_value == 'No Baseline':
        return {'display': 'none'}
    else:
        return {'display': 'block'}


@app.callback(
    Output("collapse_pca_item", "is_open"),
    [Input("collapse_button_PCA", "n_clicks")],
    [State("collapse_pca_item", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback([Output(component_id='my_slider', component_property='min'),
               Output(component_id='my_slider', component_property='max'),
               Output(component_id='my_slider', component_property='value'), ],
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'), ])

def slider_update(contents, filename):
    if contents is not None:
        df = parse_data(contents, filename)
        spectra1 = df.groupby(['Concentration(g/100g/solvents)', 'Temperature'], as_index=False).mean()
        spectra = spectra1.drop(["Concentration(g/100g/solvents)", "Temperature"], axis=1)
        wavelengths = pd.DataFrame(spectra.columns.values.astype(int))
        min = wavelengths.values.min()
        max = wavelengths.values.max()
        value = [min, max]
    else:
        min = 0
        max = 10
        value = [min, max]
    return min, max, value



@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    Output(component_id='graph_treated_data', component_property='figure'),
    Output(component_id='graph_biplot', component_property='figure'),
    Output(component_id='graph_cumvar', component_property='figure'),
    Output(component_id='graph_loadings', component_property='figure'),
    Output(component_id='graph_treated_data_PCA', component_property='figure'),
    Output('graph_PLS_predict', 'figure'),
    Output('rscv', 'children'),
    Output('rmse', 'children'),
    Output('rs', 'children'),
    Output('rmsep', 'children'),
    Output('graph_PLS_coeff', 'figure'),
    Output('graph_baseline', 'figure'),
    Output('graph_baseline_removed', 'figure'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input(component_id='my_slider', component_property='value'),
     Input(component_id="snv_checker", component_property='value'),
     Input('toggle_smooth', 'value'),
     Input("smoothing_number", 'value'),
     Input("polynomial", 'value'),
     Input("pca_checker", 'value'),
     Input('toggle_derivative', 'value'),
     Input("d_selector", 'value'),
     Input('toggle_baseline', 'value'),
     Input("bl_selector", 'value'),
     Input("d_polynomial", 'value'),
     Input("d_number", 'value'),
     Input("pc_number", 'value'),
     Input("cv_number", 'value'),
     ]
)
def data_treatments(contents, filename, wl_chosen, snv_value, toggle_smooth, S_M, P, pca_v, toggle_derivative, d_selector,toggle_baseline, bl_selector, d_p, d_n, pcn, cvn):
    if contents is not None:
        df = parse_data(contents, filename)
        spectra1 = df.groupby(['Concentration(g/100g/solvents)', 'Temperature'], as_index=False).mean()
        spectra = spectra1.drop(["Concentration(g/100g/solvents)", "Temperature"], axis=1)
        names = []
        for i in range(len(spectra1)):
            names.append("Concentration : %s Temp: %s" % ((spectra1.iloc[i, 0]), (spectra1.iloc[i, 1])))
        wavelengths = pd.DataFrame(spectra.columns.values.astype(int))
        wavelengths.columns = ['wavelength']
        spectra = spectra.T.reset_index(drop=True)
        df1 = pd.concat([wavelengths, spectra], axis=1)

        high = wavelengths[wavelengths['wavelength'] == wl_chosen[0]].index.values
        low = wavelengths[wavelengths['wavelength'] == wl_chosen[1]].index.values
        dff = df1.iloc[low[0]:high[0], :]
        dff2 = spectra.iloc[low[0]:high[0], :]

        dataPanda1 = []
        for i in range(len(dff2.columns)):
            text = names[i]
            trace = (go.Scattergl(x=dff["wavelength"], y=dff2.iloc[:, i], name=text, text=text,
                                  hoverinfo='text'))
            dataPanda1.append(trace)

        layout1 = go.Layout(
            hovermode='closest',
            title='Raman Shift: Untreated Data',
        )

        fig_raw_data = dict(data=dataPanda1, layout=layout1)

        polynomial_degree = 3  # only needed for Modpoly and IModPoly algorithm


        if toggle_baseline == 'Apply Baseline':
            baseline_data = [] 
            for i in range(len(dff2.columns)):
                Modpoly_output=[]
                baseObj = BaselineRemoval(dff2.iloc[:, i])

                if bl_selector == 'MP':
                    Modpoly_output = pd.DataFrame(baseObj.ModPoly(polynomial_degree) )
                elif bl_selector == 'IMP':
                    Modpoly_output = pd.DataFrame(baseObj.IModPoly(polynomial_degree) )
                elif bl_selector == 'ZF':
                    Modpoly_output = pd.DataFrame(baseObj.ZhangFit() )
                baseline_data.append(Modpoly_output)

            baseline_data=pd.concat(baseline_data,axis=1)
            baseline_data.reset_index()
            base = dff2.iloc[:, 1]- baseline_data.iloc[:, 1]

            fig_baseline = px.line(x=dff["wavelength"], y=dff2.iloc[:, 1])
            fig_baseline.add_trace(go.Scatter(x=dff["wavelength"], y=base, mode='lines'))
            fig_baseline.update_layout(
                title={
                    'text': "BaseLine Correction Fit",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}),

            fig_baseline.update_layout(showlegend=False)
            fig_baseline_removed = px.line(x=dff["wavelength"], y=baseline_data.iloc[:,1])
            fig_baseline_removed.update_layout(
                title={
                    'text': "Baseline Corrected",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}),

            fig_baseline_removed.update_layout(showlegend=False)
            dff2=baseline_data

        else:
            fig_baseline = go.Figure(data=[])
            fig_baseline.update_layout(
                title={
                    'text': "Baseline Correction Not Selected",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}),
            fig_baseline_removed = go.Figure(data=[])
            fig_baseline_removed.update_layout(
                title={
                    'text': "Baseline Correction Not Selected",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}),

        if snv_value == 'SNV':
            dff2 = pd.DataFrame(snv(dff2.T.values)).T

        if toggle_smooth == 'Smoothing':
            dff2 = pd.DataFrame(savgol_filter(dff2.T.values, S_M, polyorder=P, deriv=0, axis=1)).T

        if toggle_derivative == 'Apply Derivative':
            if d_selector == "FD":
                dff2 = pd.DataFrame(savgol_filter(dff2.T.values, d_n, polyorder=d_p, deriv=1, axis=1)).T
            elif d_selector == 'SD':
                dff2 = pd.DataFrame(savgol_filter(dff2.T.values, d_n, polyorder=d_p, deriv=2, axis=1)).T

        dataPanda = []

        for i in range(len(dff2.columns)):
            text = names[i]
            trace = (go.Scattergl(x=dff["wavelength"], y=dff2.iloc[:, i], name=text, text=text,
                                  hoverinfo='text'))
            dataPanda.append(trace)

        layout2 = go.Layout(
            hovermode='closest',
            title='Raman Shift: Treated Data',
        )

        fig_treated_data = dict(data=dataPanda, layout=layout2)

        xdata = (dff2.T.values)
        X = StandardScaler().fit_transform(xdata)
        pca = PCA(n_components=4)
        pca.fit_transform(X)
        pca_data = pd.DataFrame(pca.transform(X))  # get PCA coordinates for scaled_data
        per_var = pd.DataFrame(np.round(pca.explained_variance_ratio_ * 100, decimals=1), columns=['Per_Var'])
        labels_df = pd.DataFrame(['PC' + str(x) for x in range(1, len(per_var) + 1)], columns=['PC'])
        per_var_df = pd.concat([per_var, labels_df], axis=1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
        pca_data.columns = labels
        scaler = MinMaxScaler()
        loading_scores = pd.DataFrame(scaler.fit_transform((pd.Series(pca.components_[0])).values.reshape(-1, 1)))
        cum_var = pd.DataFrame(np.cumsum(per_var.values))

        if pca_v == "Temp":
            spectra1['Temperature'] = spectra1['Temperature'].astype(str)
            fig_biplot = px.scatter(pca_data, x='PC1', y='PC2', color=spectra1['Temperature'],
                               custom_data=[spectra1['Concentration(g/100g/solvents)']],
                               )
            fig_biplot.update_traces(
                hovertemplate="<br>".join([
                    "PC1: %{x}",
                    "PC2: %{y}",
                    "Concentration (g/100g solvent): %{customdata[0]}",
                    "Temperature: %{text}",

                ]),
                text=spectra1['Temperature']
            ),
            fig_biplot.update_layout(legend_title_text='Temperature')
            fig_biplot.update_traces(marker=dict(size=12))

        elif pca_v == "Conc":
            Conc = spectra1['Concentration(g/100g/solvents)']
            fig_biplot = px.scatter(pca_data, x='PC1', y='PC2', color=Conc)
            fig_biplot.update_layout(coloraxis_colorbar=dict(
                title="Concentration"))
            fig_biplot.update_traces(marker=dict(size=12))

        fig_biplot.update_layout(
            title={
                'text': "Bi-plot",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig_cumvar = px.bar(per_var_df, x='PC', y='Per_Var',
                       labels={'PC': 'Principal Component #', 'Per_Var': 'Percentage Total Variance (%)'}, color='PC')
        fig_cumvar.add_trace(go.Scatter(x=per_var_df['PC'], y=cum_var.iloc[:, 0]))
        fig_cumvar.update_layout(
            title={
                'text': "Scree Plot",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}),

        fig_cumvar.update_layout(showlegend=False)

        fig_loadings = px.line(x=dff["wavelength"], y=loading_scores.iloc[:, 0], labels=dict(x="Wavelength", y="Loadings"))

        fig_loadings.update_layout(
            title={
                'text': "Loadings Plot",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        dataPanda3 = []

        for i in range(len(dff2.columns)):
            text = names[i]
            trace = (go.Scattergl(x=dff["wavelength"], y=dff2.iloc[:, i], name=text, text=text,
                                  hoverinfo='text'))
            dataPanda3.append(trace)

        layout2 = go.Layout(
            hovermode='closest',
            title='Raman Shift: Treated Data',
            showlegend=False,
        )

        fig_data_treatment_PCA = dict(data=dataPanda3, layout=layout2)

        xdata1 = (dff2.T.values)
        X1 = StandardScaler().fit_transform(xdata1)
        y_1 = spectra1['Concentration(g/100g/solvents)']
        pls = PLSRegression(n_components=pcn)
        pls.fit(X1, y_1)
        y_cv = cross_val_predict(pls, X1, y_1, cv=cvn)
        y_c = pd.DataFrame(pls.predict(X1))
        cv = pd.concat([y_1, y_c], axis=1)
        cv.columns = ["Actual", "Predicted"]
        score_cv = round(r2_score(y_1, y_cv), 3)
        score = round(r2_score(y_1, y_c), 3)
        rmse = round((mean_squared_error(y_1, y_cv)) ** 0.5, 3)
        rmsep = round((mean_squared_error(y_1, y_c)) ** 0.5, 3)
        spectra1['Temperature'] = spectra1['Temperature'].astype(str)
        plscof = pd.DataFrame(pls.coef_[:, 0])
        dff = dff.reset_index()
        coeff = pd.concat([dff["wavelength"], plscof], axis=1)
        coeff.columns = ["wavelength", "coeffs"]
        fig_PLS_pred = px.scatter(cv, x="Predicted", y="Actual", color=spectra1['Temperature'])
        regline = sm.OLS(cv["Actual"], sm.add_constant(cv["Predicted"])).fit().fittedvalues
        fig_PLS_pred.add_traces(go.Scatter(x=cv["Predicted"], y=regline,
                                    mode='lines',
                                    marker_color='black',
                                    name='Best Fit')
                         ),
        fig_PLS_pred.update_layout(
            title={
                'text': "Predicted vs Actual Conc",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig_PLS_coeff = px.line(coeff, x="wavelength", y="coeffs")
        fig_PLS_coeff.update_layout(
            title={
                'text': "PLS Coefficients",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
    else:
        fig_raw_data = go.Figure(go.Scatter(x=1, y=1))
        fig_treated_data = go.Figure(data=[])
        fig_biplot = go.Figure(data=[])
        fig_cumvar = go.Figure(data=[])
        fig_loadings = go.Figure(data=[])
        fig_data_treatment_PCA = go.Figure(data=[])
        fig_PLS_pred = go.Figure(data=[])
        fig_PLS_coeff = go.Figure(data=[])
        fig_baseline = go.Figure(data=[])
        fig_baseline_removed = go.Figure(data=[])
        score_cv = []
        score = []
        rmse = []
        rmsep = []

    return fig_raw_data, fig_treated_data, fig_biplot, fig_cumvar, fig_loadings, fig_data_treatment_PCA, fig_PLS_pred, \
           html.H3('{0:.2f}'.format(score_cv), style={'textAlign': 'center'}), \
           html.H3('{0:.2f}'.format(rmse), style={'textAlign': 'center'}), \
           html.H3('{0:.2f}'.format(score), style={'textAlign': 'center'}), \
           html.H3('{0:.2f}'.format(rmsep), style={'textAlign': 'center'}), \
           fig_PLS_coeff, fig_baseline, fig_baseline_removed


@app.callback(
    Output("collapse_PLS_item", "is_open"),
    [Input("collapse_button_PLS", "n_clicks")],
    [State("collapse_PLS_item", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse_baseline_item", "is_open"),
    [Input("collapse_button_baseline", "n_clicks")],
    [State("collapse_baseline_item", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open



@app.callback(
    [Output(component_id='my-output', component_property='children')],
    [Input(component_id='my_slider', component_property='value')]
)
def update_graph(wl_chosen):
    return 'You have selected the range: {} - {} cm-1'.format(wl_chosen[0], wl_chosen[1]),

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False, dev_tools_props_check=False, threaded=True)

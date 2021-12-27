# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import json
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from TaskLib.task import taskMain, textTask

import base64

import db
from models import Experiment, Execution

def parse_contents(contents, filename):
    """
    Loads the input data, if it's format is JSON, otherwise throw an exception
    """
    _ , content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'json' in filename:
            json_file = json.loads(decoded)
    except Exception as e:
        print(e)
        return html.Div([
            "There was an error processing the file."
        ])
    return json_file

def get_task(task_type) -> taskMain.Task:
    """
    Maps the task_type to the corresponding Task object
    """
    #TODO get all available task
    #Similar to model's classes
    if task_type == "text-classification":
        return textTask.TextClassificationTask()

def gen_input(model_name : str, param_name : str, param_json_schema : dict):
    """
    Maps a parameter of a model to a dash input
    """
    param_type = param_json_schema.get("type")
    param_default = param_json_schema.get("default",None)

    input_component = None

    if param_type == "string":
        input_component = dcc.Dropdown(
            id=f"{model_name}-{param_name}",
            options=[{'label':opt,'value':opt} for opt in param_json_schema.get("enum")],
            value=param_default
        )

    elif param_type == "boolean":
        input_component = dcc.Dropdown(
            id=f"{model_name}-{param_name}",
            options=[{'label':opt, 'value':opt} for opt in ["True", "False"]],
            value=str(param_default)
        )
    elif param_type == "number":
        if "minimum" in param_json_schema.keys():
            input_component = dcc.Input(
                id=f"{model_name}-{param_name}",
                type="number",
                min=param_json_schema.get("minimum"),
                value=param_default
            )
        else:
            input_component = dcc.Input(
                id=f"{model_name}-{param_name}",
                type="number",
                value=param_default
            )
    elif param_type == "integer":
        input_component = dcc.Input(
            id=f"{model_name}-{param_name}",
            type="number",
            value=param_default,
            step=1,
        )

    elif param_type == "object":
        input_component = html.Div(
            id=f"params-{model_name}",
            children=[gen_input(model_name, param, param_json_schema.get("properties").get(param).get("oneOf")[0]) \
                for param in param_json_schema.get("properties").keys()]
        )
    return html.Div(
        id=f"{model_name}-{param_name}-div",
        children=[
            html.Label(f"{param_name}: "),
            input_component
        ]
    )

###################################################################################
###################################################################################

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Row([
    dbc.Col([
        dbc.Row([
        html.H2("Load Dataset"),

        # Dataset Upload
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Upload your ',
                html.A('Dataset')
            ]),
            style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'}
        ),
        html.Br(),
        # Dataset Info
        html.Div(id="dataset-info"),
        html.Br(),
        # Configure Executions
        html.Div(
            id="execution-config",
            children=[
                html.Label("Select the models to train: "),
                 html.Div(children=[
                     dcc.Checklist(
                         id='executions',
                         options=[],
                         value=[],
                     )],style={'padding': 10, 'flex': 1}),
            ],style={'display':'none'}
        )])
    ]),
    dbc.Col([
        dbc.Row([
        # Parameters
        html.Div(
            id='parameter',
            children=[
                html.Div(
                    id='exec-selection',
                    children=[
                        dcc.Dropdown(
                            id='selected-exec',
                            options=[],
                            value=[]
                        )
                    ],
                    style={'display':'none'}),
                html.Div(id='parameter-config')
            ]),
        #gen_input("SVM", "SVM", json.load(f)),
        html.Br(),
        # Experiment Results
        html.Div(id='experiment-results')
        ])
    ])
    ]),
    
    # JSON dataset
    dcc.Store(id='dataset'),
    # DB experiment id
    dcc.Store(id='experiment-id'),
    # Available models
    dcc.Store(id='available-models'),
    # Parameters dict
    dcc.Store(id='params-dict')
])
 
@app.callback([Output('dataset', 'data'),
    Output('experiment-id', 'data'),
    Output('available-models', 'data'),
    Output('dataset-info', 'children'),
    Output('executions','options'),
    Output('execution-config','style'),
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))
def load_dataset(contents, filename):
    """
    Receive the upload data input and stores in dataset.
    """
    if contents == None or filename == None:
        raise PreventUpdate

    # Get dataset
    dataset = parse_contents(contents, filename)
    dataset_info = dataset['task_info']

    dataset_information = html.H2(f"Task type: {dataset_info.get('task_type')}")

    # Create and configure task
    main_task : taskMain.Task = get_task(dataset_info['task_type'])
    main_task.config(dataset_info['task_parameters'])

    # Add experiment to DB
    exp = Experiment(**dataset_info)
    db.session.add(exp)
    db.session.commit()
    exp_id = exp.id

    # Get and show available models
    available_models = main_task.get_compatible_models()
    options=[{'label':model, 'value': model} for model in available_models]
    #childrens = [html.Label("Select the models to train: ")]
    #for model in available_models:
    #    new_children = html.Div(
    #        id=f'div-execution-{model}',
    #        children=[
    #            dcc.Checklist(
    #                id=f'execution-{model}',
    #                options=[{'label':model, 'value':model}],
    #                value=[]
    #            ),
    #            html.Button(
    #                id=f'button-{model}',
    #                value="hola"
    #            )
    #        ]
    #    )
    #    childrens.append(new_children)
    #executions = [Input(f'execution-{model}', 'value') for model in available_models]
    # Make visible the execution-config div
    style = {}
    
    return dataset, exp_id, available_models, dataset_information, options, {'style':style}

@app.callback(Output('selected-exec', 'options'),
    Output('exec-selection', 'style'),
    Input('executions', 'value'))
def enable_parameters(executions):

    if executions == []:
        raise PreventUpdate

    options=[{'label':sel_exec, 'value': sel_exec} for sel_exec in executions]
    style = {}

    return options, {'style':style}

@app.callback(Output('parameter-config', 'children'),
    Output('selected-exec', 'value'),
    Input('selected-exec', 'value'))
def load_parameter_config(selected_exec):

    if selected_exec == []:
        raise PreventUpdate
    
    f = open(f'Models/parameters/models_schemas/{selected_exec}.json')
    children = gen_input(selected_exec, selected_exec, json.load(f))
    return children, []

@app.callback(Output('experiment-results', 'children'),
    [Input('executions', 'value'),
    Input('dataset', 'data'),
    Input('experiment-id', 'data'),
    Input('params-dict', 'data')]
    #Input('available-models', 'data'),
    #*executions)
    )
def run_experiment(executions, dataset, exp_id, params_dict):

    #TODO Change this condition to a button
    #if executions == [] or dataset == None:
    raise PreventUpdate
    
    # Create and configure task
    dataset_info = dataset['task_info']
    main_task : taskMain.Task = get_task(dataset_info['task_type'])
    main_task.config(dataset_info['task_parameters'])

    # Obtain the selected models to execute
    #selected_executions = [model[0] for model in executions]
    
    params = []
    for exec in executions:
        params.append(params_dict.get(exec))

    # Set and run experiment
    main_task.set_executions(executions, params)
    main_task.run_experiments(dataset)
    
    # Store the results in the DB
    for model in main_task.experimentResults:
        exec = Execution(exp_id, model, **main_task.experimentResults[model])
        db.session.add(exec)
    db.session.commit()

    return html.Label("The experiment finish correctly")

if __name__ == '__main__':
    db.Base.metadata.create_all(db.engine)
    app.run_server(debug=True)
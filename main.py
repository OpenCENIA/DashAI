# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import json
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from TaskLib.task import taskMain, textTask

import base64

import db
from models import Experiment, Execution

app = dash.Dash(__name__)

app.layout = html.Div([
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
        'textAlign': 'center'
    }),

    html.Br(),
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
    ],style={'display':'none'}),

    html.Div(id='experiment-results'),

    dcc.Store(id='dataset'),
    dcc.Store(id='experiment-id')
])

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
 
@app.callback([Output('dataset', 'data'),
    Output('executions', 'options'),
    Output('execution-config','style'),
    Output('experiment-id', 'data')],
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
    options=[{'label':"hola", 'value': model} for model in available_models]

    # Make visible the execution-config div
    style = {}
    
    return dataset, options, {'style':style}, exp_id

@app.callback(Output('experiment-results', 'children'),
    Input('executions', 'value'),
    Input('dataset', 'data'),
    Input('experiment-id', 'data'))
def run_experiment(executions, dataset, exp_id):

    if executions == [] or dataset == None:
        raise PreventUpdate
    
    # Create and configure task
    dataset_info = dataset['task_info']
    main_task : taskMain.Task = get_task(dataset_info['task_type'])
    main_task.config(dataset_info['task_parameters'])

    # TODO Obtain params for the web app
    params = []
    for _ in executions:
        params.append({'C':[1,10]})

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
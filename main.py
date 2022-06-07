# visit http://127.0.0.1:8050/ in your web browser.

import base64
import json

import dash
import dash_bootstrap_components as dbc
from dash import ALL, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import db
from models import Execution, Experiment
from TaskLib.task.taskMain import Task
from TaskLib.task.textClassificationMLabelTask import TextClassificationMLabelTask
from TaskLib.task.textClassificationSimpleTask import TextClassificationSimpleTask


def parse_contents(contents, filename):
    """
    Loads the input data, if it's format is JSON, otherwise throw an exception
    """
    _, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "json" in filename:
            json_file = json.loads(decoded)
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing the file."])
    return json_file


def get_task(task_type) -> Task:
    """
    Maps the task_type to the corresponding Task object
    """
    # TODO get all available task
    # Similar to model's classes
    if task_type == "TextClassificationSimpleTask":
        return TextClassificationSimpleTask()
    elif task_type == "TextClassificationMLabelTask":
        return TextClassificationMLabelTask()


def gen_input(model_name: str, param_name: str, param_json_schema: dict):
    """
    Maps a parameter of a model to a dash input
    """
    param_type = param_json_schema.get("type")
    param_default = param_json_schema.get("default", None)

    input_component = None

    if param_type == "string":
        input_component = dcc.Dropdown(
            id=dict(type="form-input", name=f"{model_name}-{param_name}"),
            options=[
                {"label": opt, "value": opt} for opt in param_json_schema.get("enum")
            ],
            value=param_default,
        )

    elif param_type == "boolean":
        input_component = dcc.Dropdown(
            id=dict(type="form-input", name=f"{model_name}-{param_name}"),
            options=[{"label": opt, "value": opt} for opt in ["True", "False"]],
            value=str(param_default),
        )
    elif param_type == "number":
        if "minimum" in param_json_schema.keys():
            input_component = dcc.Input(
                id=dict(type="form-input", name=f"{model_name}-{param_name}"),
                type="number",
                min=param_json_schema.get("minimum"),
                value=param_default,
            )
        else:
            input_component = dcc.Input(
                id=dict(type="form-input", name=f"{model_name}-{param_name}"),
                type="number",
                value=param_default,
            )
    elif param_type == "integer":
        input_component = dcc.Input(
            id=dict(type="form-input", name=f"{model_name}-{param_name}"),
            type="number",
            value=param_default,
            step=1,
        )

    elif param_type == "object":
        input_component = html.Div(
            id=dict(type="form-input", name=f"{model_name}-{param_name}"),
            children=[
                gen_input(
                    model_name,
                    param,
                    param_json_schema.get("properties").get(param).get("oneOf")[0],
                )
                for param in param_json_schema.get("properties").keys()
            ],
        )
    return html.Div(
        id=f"{model_name}-{param_name}-div",
        children=[
            html.Label(
                f"{param_name}: ",
                id=dict(type="form-label", name=f"{model_name}-{param_name}"),
            ),
            input_component,
        ],
    )


###############################################################################
###############################################################################

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        dbc.Row(
            [
                html.Div(
                    id="test-parameters",
                    children=[html.H2("Muestra los parámetros")],
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                html.H2("Load Dataset"),
                                # Dataset Upload
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Upload your ", html.A("Dataset")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                    },
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
                                        html.Div(
                                            children=[
                                                dcc.Checklist(
                                                    id="executions",
                                                    options=[],
                                                    value=[],
                                                )
                                            ],
                                            style={"padding": 10, "flex": 1},
                                        ),
                                    ],
                                    style={"display": "none"},
                                ),
                            ]
                        )
                    ]
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                # Parameters
                                html.Div(
                                    id="parameter",
                                    children=[
                                        html.Div(
                                            id="exec-selection",
                                            children=[
                                                dcc.Dropdown(
                                                    id="selected-exec",
                                                    options=[],
                                                    value=[],
                                                )
                                            ],
                                            style={"display": "none"},
                                        ),
                                        html.Div(id="parameter-config"),
                                        dbc.Button(
                                            "Submit",
                                            color="dark",
                                            className="me-1",
                                            id="button-submit",
                                        ),
                                    ],
                                ),
                                # gen_input("SVM", "SVM", json.load(f)),
                                html.Br(),
                                # Experiment Results
                                html.Div(id="experiment-results"),
                            ]
                        )
                    ]
                ),
            ]
        ),
        # JSON dataset
        dcc.Store(id="dataset"),
        # DB experiment id
        dcc.Store(id="experiment-id"),
        # Available models
        dcc.Store(id="available-models"),
        # Parameters dict
        dcc.Store(id="params-dict"),
    ]
)


@app.callback(
    Output("test-parameters", "children"),
    [Input("button-submit", component_property="n_clicks")],
    [
        State(dict(type="form-input", name=ALL), "value"),
        State(dict(type="form-label", name=ALL), "children"),
    ],
)
def get_parameters(n_clicks, values, labels):
    if n_clicks is None:
        raise PreventUpdate
    params_dict = {}
    for label, value in zip(labels[1:], values[1:]):
        params_dict[label[:-2]] = value
    return str(params_dict)


@app.callback(
    [
        Output("dataset", "data"),
        Output("experiment-id", "data"),
        Output("dataset-info", "children"),
        Output("executions", "options"),
        Output("execution-config", "style"),
    ],
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def load_dataset(contents, filename):
    """
    Receive the upload data input and stores in dataset.
    Obtain the available models and shows to the user.
    """
    if contents is None or filename is None:
        raise PreventUpdate

    # Get dataset
    dataset: dict = parse_contents(contents, filename)
    dataset_info: dict = dataset.get("task_info")
    task_type: str = dataset_info.get("task_type")

    dataset_information = html.H2(f"Task type: {task_type}")

    # Create and configure task
    main_task: Task = get_task(task_type)

    # Add experiment to DB
    exp = Experiment(**dataset_info)
    db.session.add(exp)
    db.session.commit()
    exp_id = exp.id

    # Get and show available models
    available_models: list = main_task.get_compatible_models()
    options: list = [{"label": model, "value": model} for model in available_models]

    # Make visible the execution-config div
    style = {}

    return dataset, exp_id, dataset_information, options, {"style": style}


@app.callback(
    Output("selected-exec", "options"),
    Output("exec-selection", "style"),
    Input("executions", "value"),
)
def enable_parameters(executions):
    """
    Get the selected models, and load a dropdown list to select any and
    configure its parameters.
    """
    if executions == []:
        raise PreventUpdate

    options = [{"label": sel_exec, "value": sel_exec} for sel_exec in executions]
    style = {}

    return options, {"style": style}


@app.callback(
    Output("parameter-config", "children"),
    Output("selected-exec", "value"),
    Input("selected-exec", "value"),
)
def load_parameter_config(selected_exec):
    """
    Get the selected execution and loads its parameters menu
    """
    if selected_exec == []:
        raise PreventUpdate

    f = open(f"Models/parameters/models_schemas/{selected_exec}.json")
    children = gen_input(selected_exec, selected_exec, json.load(f))
    return children, []


@app.callback(Output("params-dict", "data"), Input("algo", "algo"))
def dummy(algo=None):
    return {}


@app.callback(
    Output("experiment-results", "children"),
    [
        Input("executions", "value"),
        Input("dataset", "data"),
        Input("experiment-id", "data"),
        Input("params-dict", "data"),
    ],
)
def run_experiment(executions, dataset, exp_id, params_dict):

    # TODO Change this condition to a button
    # if executions == [] or dataset == None:
    raise PreventUpdate

    # Create and configure task
    dataset_info = dataset["task_info"]
    main_task: Task = get_task(dataset_info["task_type"])
    main_task.config(dataset_info["task_parameters"])

    # Obtain the selected models to execute
    # selected_executions = [model[0] for model in executions]

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


if __name__ == "__main__":
    db.Base.metadata.create_all(db.engine)
    app.run_server(debug=True)

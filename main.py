# visit http://127.0.0.1:8050/ in your web browser.

import base64
import json

import dash
import dash_bootstrap_components as dbc
from dash import ALL, callback_context, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import db
from models import Experiment
from Models.classes.getters import filter_by_parent
from TaskLib.task.numericClassificationTask import NumericClassificationTask
from TaskLib.task.taskMain import Task
from TaskLib.task.textClassificationTask import TextClassificationTask

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)


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
    if task_type == "TextClassificationTask":
        return TextClassificationTask()
    elif task_type == "NumericClassificationTask":
        return NumericClassificationTask()
    else:
        Exception("Task not recognized")


def gen_input(
    model_name: str,
    param_name: str,
    param_json_schema: dict,
    parent_model_data: dict = None,
):
    """
    Maps a parameter of a model to a dash input
    """
    param_type = param_json_schema.get("type")
    param_default = param_json_schema.get("default", None)
    input_component = None

    id_dict = dict(type="form-input", name=f"{model_name}--{param_name}")
    label_id_dict = dict(type="form-label", name=f"{model_name}--{param_name}")
    if parent_model_data is not None:
        id_dict["model"] = parent_model_data["model"]
        id_dict["model_parameter"] = parent_model_data["parameter"]
        label_id_dict["model"] = parent_model_data["model"]
        label_id_dict["model_parameter"] = parent_model_data["parameter"]

    if param_type == "string":
        input_component = dcc.Dropdown(
            id=id_dict,
            options=[
                {"label": opt, "value": opt} for opt in param_json_schema.get("enum")
            ],
            value=param_default,
        )

    elif param_type == "boolean":
        input_component = dcc.Dropdown(
            id=id_dict,
            options=[{"label": opt, "value": opt} for opt in ["True", "False"]],
            value=str(param_default),
        )
    elif param_type == "number":
        if "minimum" in param_json_schema.keys():
            input_component = dcc.Input(
                id=id_dict,
                type="number",
                min=param_json_schema.get("minimum"),
                value=param_default,
            )
        else:
            input_component = dcc.Input(
                id=id_dict,
                type="number",
                value=param_default,
            )
    elif param_type == "integer":
        input_component = dcc.Input(
            id=id_dict,
            type="number",
            value=param_default,
            step=1,
        )

    elif param_type == "object":
        input_component = html.Div(
            id=id_dict,
            children=[
                gen_input(
                    model_name,
                    param,
                    param_json_schema.get("properties").get(param).get("oneOf")[0],
                    parent_model_data=parent_model_data,
                )
                for param in param_json_schema.get("properties").keys()
            ],
        )

    elif param_type == "class":
        parent_class_name = param_json_schema.get("parent")
        input_component = html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id=dict(
                                type="recursive-parameter-dropdown",
                                name=f"{model_name}--{param_name}",
                            ),
                            options=[
                                {"label": opt, "value": opt}
                                for opt in filter_by_parent(parent_class_name).keys()
                            ],
                            value=str(param_default),
                        )
                    ]
                ),
                dbc.Accordion(
                    id=dict(
                        type="recursive-parameter-accordion",
                        name=f"{model_name}--{param_name}",
                    ),
                    children=[],
                    flush=True,
                ),
            ]
        )

    return html.Div(
        id=f"{model_name}-{param_name}-div",
        children=[html.Label(f"{param_name}: ", id=label_id_dict), input_component],
    )


###################################################################################
###################################################################################

app.layout = html.Div(
    [
        dbc.Row(
            [  # html.Div(id='test-parameters', children = [html.H2()]),
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
                                # models_table(),
                                # parameter_config_modal()
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
                                            id="test-parameters",
                                            children=[
                                                html.H2("Muestra los parámetros")
                                            ],
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
                                                                [
                                                                    "Upload your ",
                                                                    html.A("Dataset"),
                                                                ]
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
                                                                html.Label(
                                                                    "Select the models"
                                                                    "to train: "
                                                                ),
                                                                html.Div(
                                                                    children=[
                                                                        dcc.Checklist(
                                                                            id="exe"
                                                                            "cutions",
                                                                            options=[],
                                                                            value=[],
                                                                        )
                                                                    ],
                                                                    style={
                                                                        "padding": 10,
                                                                        "flex": 1,
                                                                    },
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
                                                                            id="select"
                                                                            "ed-exec",
                                                                            options=[],
                                                                            value=[],
                                                                        )
                                                                    ],
                                                                    style={
                                                                        "display": ""
                                                                        "none"
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    id="parameter"
                                                                    "-config"
                                                                ),
                                                                dbc.Button(
                                                                    "Submit",
                                                                    color="dark",
                                                                    className="me-1",
                                                                    id="button-submit",
                                                                ),
                                                            ],
                                                        ),
                                                        html.Br(),
                                                        # Experiment Results
                                                        html.Div(
                                                            id="experiment-results"
                                                        ),
                                                    ]
                                                )
                                            ],
                                            style={"display": "none"},
                                        ),
                                        html.Div(id="parameter-config"),
                                        html.Br(),
                                        dbc.Button(
                                            "Save",
                                            color="dark",
                                            className="me-1",
                                            id="button-submit",
                                            style={"display": "none"},
                                        ),
                                        html.Br(),
                                        dbc.Button(
                                            "Run Experiment",
                                            color="dark",
                                            className="me-1",
                                            n_clicks=0,
                                            id="button-run-experiment",
                                            style={"display": "none"},
                                        ),
                                    ],
                                ),
                                # gen_input("SVM", "SVM", json.load(f)),
                                html.Br(),
                                # Experiment Results
                                html.Div(id="experiment-results"),
                                html.Div(id="test-test", children=[html.H2()]),
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
    Output(dict(type="recursive-parameter-accordion", name=ALL), "children"),
    [
        Input(dict(type="recursive-parameter-dropdown", name=ALL), "value"),
        Input(dict(type="recursive-parameter-dropdown", name=ALL), "id"),
    ],
    State(dict(type="recursive-parameter-accordion", name=ALL), "children"),
    prevent_initial_call=True,
)
def display_recursive_parameter_form(
    rec_parameter_selected_options,
    rec_parameter_dropdown_ids,
    prev_rec_parameters_forms,
):
    """
    Get the selected option for all recursive parameters dropdowns and display
    the forms to configure each one of them.
    """
    # dictionary to map the names from the dropdown and the names in the json schema
    tokenizer_map = {"NormalTokenizer": "normalTok", "TweetTokenizer": "tweetTok"}

    # get the component that triggered the callback
    triggered = callback_context.triggered[-1]["prop_id"]
    triggered_id = json.loads(triggered.split(".")[0])

    for i, (option, id) in enumerate(
        zip(rec_parameter_selected_options, rec_parameter_dropdown_ids)
    ):
        # update only the form of the component that triggered the callback
        if id == triggered_id:
            mapped_option = (
                tokenizer_map[option] if option in tokenizer_map.keys() else option
            )
            f = open(f"Models/parameters/models_schemas/{mapped_option}.json")
            name = id["name"].split("--")
            accordion_item_form = dbc.AccordionItem(
                [
                    gen_input(
                        mapped_option,
                        mapped_option,
                        json.load(f),
                        parent_model_data={"model": name[0], "parameter": name[1]},
                    )
                ],
                title=f"{mapped_option} parameters",
            )
            prev_rec_parameters_forms[i] = accordion_item_form

    return prev_rec_parameters_forms


@app.callback(
    Output("params-dict", "data"),
    # Output('test-parameters', 'children'),
    [
        Input("button-submit", component_property="n_clicks"),
        Input("executions", "value"),
    ],
    [
        State(dict(type="form-input", name=ALL), "value"),
        State(dict(type="form-label", name=ALL), "children"),
        State(
            dict(type="form-input", model=ALL, model_parameter=ALL, name=ALL), "value"
        ),
        State(
            dict(type="form-label", model=ALL, model_parameter=ALL, name=ALL),
            "children",
        ),
        State(dict(type="form-input", model=ALL, model_parameter=ALL, name=ALL), "id"),
        State("params-dict", "data"),
    ],
    prevent_initial_call=True,
)
def store_parameters(
    n_clicks,
    executions,
    values,
    labels,
    rec_params_values,
    rec_params_labels,
    rec_params_ids,
    prev_params_dict,
):
    """
    Get and store default parameters when a model is selected.
    Retrieve and store parameters configured by the user.
    """

    # Store in "params-dict" the default values for the parameters
    # of each selected model.
    if callback_context.triggered[0]["prop_id"] == "executions.value":
        default_params_dict = {}
        for sel_exec in executions:
            sel_exec_default_params = {}
            with open(f"Models/parameters/models_schemas/{sel_exec}.json") as f:
                param_json_schema = json.load(f)
                for param in param_json_schema.get("properties").keys():
                    sel_exec_default_params[param] = (
                        param_json_schema.get("properties")
                        .get(param)
                        .get("oneOf")[0]
                        .get("default")
                    )
            default_params_dict[sel_exec] = sel_exec_default_params
        return default_params_dict  # , str(default_params_dict)

    # Retrieve and store parameters defined by the user when "Save" button is pressed.
    elif callback_context.triggered[0]["prop_id"] == "button-submit.n_clicks":
        params_dict = {}
        for label, value in zip(labels[1:], values[1:]):
            params_dict[label[:-2]] = value
        output_params_dict = {}
        if prev_params_dict is None:
            output_params_dict = {labels[0][:-2]: params_dict}
        else:
            prev_params_dict[labels[0][:-2]] = params_dict
            output_params_dict = prev_params_dict

        # recursive parameters

        if (
            len(rec_params_values) > 0
            and len(rec_params_ids) > 0
            and len(rec_params_labels) > 0
        ):
            rec_parameters_dict = {}
            rec_param_name = rec_params_labels[0][:-2]
            model = rec_params_ids[0]["model"]
            param_name_in_model = rec_params_ids[0]["model_parameter"]

            for id, value, label in zip(
                rec_params_ids, rec_params_values, rec_params_labels
            ):
                name = id["name"].split("--")
                # this condition refers to when "gen_input" creates a label for the
                # model as it was another parameter, then "model_name" (name[0]) has
                # the same value as "param_name" (name[1]) and that is used to mark
                # the first parameter of a recursive parameter form.
                if len(name) == 2 and name[0] == name[1]:
                    # if rec_parameters_dict is not an empty dict ({})
                    if rec_parameters_dict:
                        output_params_dict[model][param_name_in_model] = {
                            "value": rec_param_name,
                            "parameters": rec_parameters_dict,
                        }

                    # resets the values to add the next recursive parameter form.
                    rec_parameters_dict = {}
                    rec_param_name = name[0]
                    param_name_in_model = id["model_parameter"]
                    continue

                rec_parameters_dict[label[:-2]] = value

            # last recursive parameter form is never added in the previous
            # for loop so it needs to be added after the for loop ends.
            output_params_dict[model][param_name_in_model] = {
                "value": rec_param_name,
                "parameters": rec_parameters_dict,
            }

        return output_params_dict  # , str(output_params_dict)


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
    # task_type: str = dataset_info.get("task_type")

    # Create and configure task
    # main_task: Task = get_task(task_type)
    # Add experiment to DB
    exp = Experiment(**dataset_info)
    db.session.add(exp)
    db.session.commit()

    # Get and show available models
    # available_models: list = main_task.get_compatible_models()
    # options: list = [{"label": model, "value": model} for model in available_models]

    # Make visible the execution-config div
    # style = {}


@app.callback(
    Output("selected-exec", "options"),
    Output("exec-selection", "style"),
    Output("button-run-experiment", "style"),
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

    return options, {"style": style}, {"style": style}


@app.callback(
    Output("parameter-config", "children"),
    Output("selected-exec", "value"),
    Output("button-submit", "style"),
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
    return children, [], {"style": {}}


# @app.callback(Output('params-dict', 'data'),
#    Input('algo', 'algo'))
# def dummy(algo=None):
#    return {}


@app.callback(
    Output("experiment-results", "children"),
    [
        Input("button-run-experiment", "n_clicks"),
        Input("executions", "value"),
        Input("dataset", "data"),
        Input("experiment-id", "data"),
        Input("params-dict", "data"),
    ],
    prevent_initial_call=True,
)
def run_experiment(n_clicks, executions, dataset, exp_id, params_dict):
    # TODO Change this condition to a button
    # if executions == [] or dataset == None:
    if n_clicks == 0:
        raise PreventUpdate

    # Create and configure task
    dataset_info = dataset["task_info"]
    main_task: Task = get_task(dataset_info["task_type"])

    # Set and run experiment
    main_task.set_executions(executions, params_dict)
    main_task.run_experiments(dataset)

    # Store the results in the DB
    for model in main_task.experimentResults:
        print(main_task.experimentResults[model])
    #    exec = Execution(exp_id, model, **main_task.experimentResults[model])
    #    db.session.add(exec)
    # db.session.commit()

    return html.Label("The experiment finish correctly")


if __name__ == "__main__":
    db.Base.metadata.create_all(db.engine)
    app.run_server(debug=True)

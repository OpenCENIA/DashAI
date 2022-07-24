# Run this app with `python app.py` and
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
    level: int = 0
):
    """
    Maps a parameter of a model to a dash input
    """
    param_type = param_json_schema.get("type")
    param_default = param_json_schema.get("default", None)
    input_component = None

    id_dict = dict(type="form-input", name=f"{model_name}--{param_name}", level=level)
    label_id_dict = dict(type="form-label", name=f"{model_name}--{param_name}", level=level)

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
            input_component = dcc.Input(id=id_dict, type="number", value=param_default)
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
                    level=level,
                )
                for param in param_json_schema.get("properties").keys()
            ],
        )

    elif param_type == "class":
        label_id_dict["type"] = "recursive-parameter-label"
        parent_class_name = param_json_schema.get("parent")
        input_component = html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Dropdown(
                            id=dict(
                                type="recursive-parameter-dropdown",
                                name=f"{model_name}--{param_name}",
                                level=level
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
            [
                html.Div(id='test-parameters', children = [html.H2()]),
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


#dictionary to map the names from the dropdown and the names in the json schema
#used in the following callbacks: - display_recursive_parameter_form
#                                 - store_parameters
##############################################################################
tokenizer_map = {"NormalTokenizer": "normalTok", "TweetTokenizer": "tweetTok"}
##############################################################################

@app.callback(
    Output(dict(type="recursive-parameter-accordion", name=ALL), "children"),
    [
        Input(dict(type="recursive-parameter-dropdown", name=ALL, level=ALL), "value"),
        Input(dict(type="recursive-parameter-dropdown", name=ALL, level=ALL), "id"),
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
    #dictionary to map the names from the dropdown and the names in the json schema
    #tokenizer_map = {"NormalTokenizer": "normalTok", "TweetTokenizer": "tweetTok"}

    # get the component that triggered the callback
    triggered = callback_context.triggered[-1]["prop_id"]
    triggered_id = json.loads(triggered.split(".")[0])

    for i, (option, id) in enumerate(
        zip(rec_parameter_selected_options, rec_parameter_dropdown_ids)
    ):
        # update only the form of the component that triggered the callback
        if id == triggered_id:

            try:
                string_int = id["level"]
                level = int(string_int)
            except ValueError:
                print(f"{string_int} in id: {id} is not an integer")
                break

            mapped_option = (
                tokenizer_map[option] if option in tokenizer_map.keys() else option
            )
            f = open(f"Models/parameters/models_schemas/{mapped_option}.json")
            accordion_item_form = dbc.AccordionItem(
                [
                    gen_input(
                        mapped_option,
                        mapped_option,
                        json.load(f),
                        level=level+1
                    )
                ],
                title=f"{mapped_option} parameters",
            )
    

            prev_rec_parameters_forms[i] = accordion_item_form

    return prev_rec_parameters_forms


@app.callback(
    Output('params-dict', 'data'),
    # Output('test-parameters', 'children'),
    [
        Input('button-submit', component_property='n_clicks'),
        Input('executions', 'value')
    ],
    [
        State(
            dict(type='form-input', name=ALL, level=ALL), 'value'
        ),
        State(
            dict(type='form-label', name=ALL, level=ALL), 'children'
        ),
        State(
            dict(type='form-input', name=ALL, level=ALL), 'id'
        ),
        State(
            dict(type="recursive-parameter-dropdown", name=ALL, level=ALL), 'value'
        ),
        State(
            dict(type="recursive-parameter-label", name=ALL, level=ALL), 'children'
        ),
        State(
            dict(type="recursive-parameter-dropdown", name=ALL, level=ALL), 'id'
        ),
        State('params-dict', 'data')
    ],
    prevent_initial_call=True
)
def store_parameters(
    n_clicks,
    executions,
    values,
    labels,
    ids,
    rec_params_values,
    rec_params_labels,
    rec_params_ids,
    prev_params_dict
):
    """
    Get and store default parameters when a model is selected.
    Retrieve and store parameters configured by the user.
    """

    def is_recursive_parameter(label: str):
        return label in rec_params_labels
    
    #Store in "params-dict" the default values for the parameters of each selected model
    #when the element that triggered the callback is the executions dropdown
    if callback_context.triggered[-1]['prop_id'] == 'executions.value':
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
        return default_params_dict#, str(default_params_dict)

    #Retrieve and store parameters selected by the user when "Save" button is pressed.
    #when the element that triggered the callback is the "Save" button
    elif callback_context.triggered[-1]['prop_id'] == 'button-submit.n_clicks':
        
        # merge the lists of normal and recursive paramteters 
        values += rec_params_values
        labels += rec_params_labels
        ids += rec_params_ids

        #parse parameter lists (values, labels, ids)
        levels = {}
        for value, label, id in zip(values, labels, ids):
            try:
                string_int = id["level"]
                level = int(string_int)
            except ValueError:
                print(f"{string_int} in id: {id} is not an integer")
                break

            try:
                model, param = id["name"].split("--")
            except ValueError:
                name = id["name"]
                print(f"name: {name} from id: {id} is not of the form <model>--<parameter>")
                break
            
            if model != param:
                param_data = (label, value, model, param)
                if level in levels.keys():
                    levels[level].append(param_data)
                else:
                    levels[level] = [param_data]
            else:
                continue

        #construction of parameters dict
        previous_level_parameters = {}
        for level in sorted(levels.keys(), reverse = True):
            parameters = {}
            for label, value, model, param in levels[level]:
                if is_recursive_parameter(label):
                        mapped_value = tokenizer_map[value] if value in tokenizer_map.keys() else value
                        value = previous_level_parameters[mapped_value] if mapped_value != 'None' else None

                if level > 0:
                    if model in parameters.keys():
                        parameters[model]["parameters"][param] = value
                    else:
                        parameters[model] = {"value": model, "parameters": {param: value}}
                else:
                    parameters[param] = value
            previous_level_parameters  = parameters

        #assuming every parameter in level 0 has the model (e.g NumericalWrapperForText) as its parent        
        model_name = levels[0][0][2]

        prev_params_dict = {} if prev_params_dict is None else prev_params_dict

        prev_params_dict[model_name] = parameters

        return prev_params_dict#, str(prev_params_dict)       
        
@app.callback([Output('dataset', 'data'),
    Output('experiment-id', 'data'),
    Output('dataset-info', 'children'),
    Output('executions','options'),
    Output('execution-config','style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))
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

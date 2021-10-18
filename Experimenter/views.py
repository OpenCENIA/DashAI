#from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from .models import Execution, Experiment
#from .models import Number, Execution
#from Models.classes import *
#from Models.metrics import *

from TaskLib.task.taskMain import Task
from TaskLib.task.textTask import TextTask
from Models.classes import *

#from datetime import datetime
import json
#import numpy

utils = {}

def index(request):
    return render(request, 'Experimenter/index.html')

def init_experiment(request):
    return render(request, "Experimenter/conf_task.html")

def configure_experiment(request):
    """
    This view recieves the information about the task, creating it.
    Then redirect to the configure_models view to select the models to train and
    its parameters.
    """
    dataset = json.loads(request.FILES['DATASET'].read())
    request.session["DATASET"] = dataset

    task_type = request.POST.get("TASK")
    task : Task = obtain_task(task_type)
    task.config(request.POST)
    utils["TASK"] = task

    e = Experiment(
        task_type = task_type,
        task_parameters = json.dumps(task.get_parameters())
    )
    e.save()
    utils["EXP"] = e

    return render(request, 'Experimenter/make_experiment.html', {'classes': task.get_compatible_models()})

def execute_experiment(request):
    """
    This view recieves the information about the models and its parameters.
    The performs the training of all models and stores it in the DB.
    """
    task : Task = utils["TASK"]

    models = [] # Obtain the name of all selected models
    for model_name in task.get_compatible_models():
        if (model_name + "_checkbox") in request.POST:
            models += [request.POST[model_name + "_checkbox"]]
    #TODO obtain the parameters
    params = []
    for model_name in models:
        params.append({'C':[1,10]})
    
    task.set_executions(models, params)

    task.run_experiments(request.session["DATASET"])

    for exec_name in task.experimentResults:
        Execution.objects.create(
            experiment = utils["EXP"],
            execution_model = exec_name,
            execution_file = task.experimentResults[exec_name]["executionBytes"],
            parameters = task.experimentResults[exec_name]["parameters"],
            train_results = task.experimentResults[exec_name]["train_results"],
            test_results = task.experimentResults[exec_name]["test_results"]
        )
    return redirect('/experimenter')

# def make_experiment(request):
#     classes = get_available_models()
#     metrics = get_available_metrics()
#     return render(request, 'Experimenter/make_experiment.html', {'classes': classes, 'metrics': metrics})

# def run_experiment(request):

#     dataset = json.loads(request.FILES['dataset'].read())

#     train_partition = process_data(dataset.get('train'))
#     val_partition = process_data(dataset.get('val'))
#     test_partition = process_data(dataset.get('test'))

#     experiment_name = "Exp_" + datetime.now().strftime("%H:%M:%S")

#     experiment_confs = {}
#     experiment_confs['label_conf'] = request.POST.get('label_atr')
#     experiment_confs['doc_conf'] = request.POST.get('doc_atr')

    
    
#     metrics = [] # Obtain the name of all selected metrics
#     for metric_name in get_available_metrics():
#         if (metric_name + "_checkbox") in request.POST:
#             metrics += [request.POST[metric_name + "_checkbox"]]

#     for model in models:
#         execution = globals().get(model)()
#         execution.fit(execution.preprocess(train_partition[0]), train_partition[1])

#         model_confs = experiment_confs
#         model_confs['model_conf'] = execution.params

#         model_result = {}
#         predicted_labels = execution.predict(execution.preprocess(test_partition[0]))
#         for metric in metrics:
#             model_result[metric] = globals().get(metric)()(test_partition[1].transpose(), predicted_labels.transpose())
        
#         Execution.objects.create(
#             experiment_name= experiment_name,
#             execution_file= execution.save(),
#             configurations= model_confs,
#             results= model_result
#             )
    
#     return redirect('/experimenter')
    

def test(request):
    print(Execution.objects.get(id=1).results)
    return render(request, 'Experimenter/test.html')

def obtain_task(task_type) -> Task:
    if task_type == "TEXT":
        return TextTask()

# def process_data(data_dict):
#     # Dict(Str,Int) -> List(NumpyArray(Str),NumpyArray(Int))
#     # This function takes a dataset dictionary (train, val or test) and
#     # produces a numpy array with de text list and the labels list
#     text_list,labels_list = [], []
#     for example in data_dict:
#         text_list += [example['text']]
#         labels_list += [example['label']]
#     return [numpy.array(text_list), numpy.array(labels_list)]

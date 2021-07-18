from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from .models import Number, Execution
from Models.classes import *
from Models.metrics import *

from datetime import datetime
import json
import numpy

def index(request):
    return render(request, 'Experimenter/index.html')

def make_experiment(request):
    classes = get_available_models()
    metrics = get_available_metrics()
    return render(request, 'Experimenter/make_experiment.html', {'classes': classes, 'metrics': metrics})

def run_experiment(request):

    dataset = json.loads(request.FILES['dataset'].read())

    train_partition = process_data(dataset.get('train'))
    val_partition = process_data(dataset.get('val'))
    test_partition = process_data(dataset.get('test'))

    experiment_name = "Exp_" + datetime.now().strftime("%H:%M:%S")

    experiment_confs = {}
    experiment_confs['label_conf'] = request.POST.get('label_atr')
    experiment_confs['doc_conf'] = request.POST.get('doc_atr')

    models = [] # Obtain the name of all selected models
    for model_name in get_available_models():
        if (model_name + "_checkbox") in request.POST:
            models += [request.POST[model_name + "_checkbox"]]
    
    metrics = [] # Obtain the name of all selected metrics
    for metric_name in get_available_metrics():
        if (metric_name + "_checkbox") in request.POST:
            metrics += [request.POST[metric_name + "_checkbox"]]

    for model in models:
        execution = globals().get(model)()
        execution.fit(execution.preprocess(train_partition[0]), train_partition[1])

        model_confs = experiment_confs
        model_confs['model_conf'] = execution.params

        model_result = {}
        predicted_labels = execution.predict(execution.preprocess(test_partition[0]))
        for metric in metrics:
            model_result[metric] = globals().get(metric)()(test_partition[1].transpose(), predicted_labels.transpose())
        
        Execution.objects.create(
            experiment_name= experiment_name,
            execution_file= execution.save(),
            configurations= model_confs,
            results= model_result
            )
    
    return redirect('/experimenter')
    

def test(request):
    print(Execution.objects.get(id=1).results)
    return render(request, 'Experimenter/test.html')

def process_data(data_dict):
    # Dict(Str,Int) -> List(NumpyArray(Str),NumpyArray(Int))
    # This function takes a dataset dictionary (train, val or test) and
    # produces a numpy array with de text list and the labels list
    text_list,labels_list = [], []
    for example in data_dict:
        text_list += [example['text']]
        labels_list += [example['label']]
    return [numpy.array(text_list), numpy.array(labels_list)]

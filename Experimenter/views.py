from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from numpy.lib.polynomial import _polyder_dispatcher
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
    models = []
    for model_name in get_available_models():
        if (model_name + "_checkbox") in request.POST:
            models += [request.POST[model_name + "_checkbox"]]
    
    metrics = []
    for metric_name in get_available_metrics():
        if (metric_name + "_checkbox") in request.POST:
            metrics += [request.POST[metric_name + "_checkbox"]]

    dataset = json.loads(request.FILES['dataset'].read())
    train_partition = [[], []]
    cont = 0
    for train_dict in dataset.get('train'):
        if cont == 10000:
            break
        else:
            cont += 1
        train_partition = [
            train_partition[0]+[train_dict.get('text')], 
            train_partition[1]+[train_dict.get('label')]
            ]
    train_partition = [numpy.array(train_partition[0]), numpy.array(train_partition[1])]
    
    '''
    val_partition = [[], []]
    for val_dict in dataset.get('val'):
        val_partition = [
            val_partition[0]+[val_dict.get('text')], 
            val_partition[1]+[val_dict.get('label')]
            ]
    '''
    
    test_partition = [[], []]
    cont = 0
    for test_dict in dataset.get('test'):
        if cont == 10000:
            break
        else:
            cont += 1
        test_partition = [
            test_partition[0]+[test_dict.get('text')], 
            test_partition[1]+[test_dict.get('label')]
            ]
    test_partition = [numpy.array(test_partition[0]), numpy.array(test_partition[1])]

    experiment_name = "Exp_" + datetime.now().strftime("%H:%M:%S")
    dataset_label_atr = request.POST.get('label_atr')
    dataset_doc_atr = request.POST.get('doc_atr')

    for model in models:
        execution = globals().get(model)()
        execution.fit(execution.preprocess(train_partition[0]), train_partition[1])
        
        results = {}
        predicted_labels = execution.predict(execution.preprocess(test_partition[0]))
        for metric in metrics:
            results[metric] = globals().get(metric)()(test_partition[1].transpose(), predicted_labels.transpose())
        
        Execution.objects.create(
            experiment_name= experiment_name,
            execution_file= execution.save(),
            configurations= execution.params,
            results= results
            )
        print(results)
    
    return redirect('/experimenter')
    

def test(request):
    print(Execution.objects.get(id=1).results)
    return render(request, 'Experimenter/test.html')

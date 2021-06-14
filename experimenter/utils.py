"""
Módulo experimentador:
Contiene todo lo necesario para correr un experimento de la api.
"""
import logging
from datetime import datetime

import jsonschema
from sklearn.model_selection import ParameterGrid

from parameters.experimenter_schema import get_full_schema
#from parameters.metrics_mapping import metrics_mapping
#from parameters.models_mapping import models_mapping
from parameters.preprocesses_mapping import preprocesses_mapping
from parameters.tokenizers_mapping import tokenizers_mapping
import models

get_models_mapping = models.arquitectures_mapping
get_metrics_mapping = models.metrics_mapping

logger = logging.getLogger()

def get_dataset(dataset_dic):
    if "name" in dataset_dic.keys() :
        dataset = models.Dataset.query.filter_by(name=dataset_dic['name']).first()
    else:
        dataset = models.Dataset.query.get(dataset_dic['id'])
    return dataset

def check_experimenter_input(js):
    """
    Valida el input de entrada.

    Args:
        js (dict): Diccionario con el json parseado.
    """
    experimenter_schema = get_full_schema()
    try:
        jsonschema.validate(js, experimenter_schema)
    except jsonschema.ValidationError as err:
        return models.VerificationResult(False, err.schema['error_msg'])
    dataset = get_dataset(js["datasets"])
    task = dataset.tags["task"]
    if js["task"] != task:
        return models.VerificationResult(False, f'La task entregada es distinta con la del dataset: \n{js["task"]} no es {task}')
    return models.VerificationResult(True)

def creator(models_dic, metrics_dic):
    """
    Llama a las funciones creadoras de modelos y métricas.

    Args:
        models_dic (dict): Diccionario de modelos ingresados.
        metrics_dic (dict): Diccionario de métricas ingresadas.

    Returns:
        tuple: Tupla cuyo primer elemento corresponde a la lista de modelos
                instanciados, y el segundo elemento es una lista de métricas
                instanciadas.
    """
    return models_creator(models_dic), metrics_creator(metrics_dic)

def filter_models(ranked_models, n):
    """
    Retorna los n primeros reportes de modelos.

    Args:
        ranked_models (list): Reportes de modelos ordenados.
        n (int): Cantidad de modelos a escoger.

    Returns:
        list: Lista de n reportes de modelos ordenados.
    """
    if n == 0:
        return [[model['model'] for model in report]
                for report in ranked_models]
    return [[model['model'] for model in report[:n]]
                for report in ranked_models]

def get_optimizer_params(metrics_dic):
    """
    Función auxiliar, para obtener directamente los parámetros
    del optimizador.

    Args:
        metrics_dic (dict): Diccionario de métricas entregado en el input.

    Returns:
        tuple: Tupla de tres elementos. El primero corresponde a la label
                según la cual se optimizará, mientras que el segundo es la
                métrica con respecto a la que se realizará dicha optimización.
                El tercero corresponde al número de modelos que se quiere
                rescatar.
    """
    opt_label = str(metrics_dic['optimizer_label'])
    opt_metric = metrics_dic['optimizer_metric']
    best_n = metrics_dic['best_n']
    return opt_label, opt_metric, best_n

def metrics_creator(list_metrics):
    """
    Entrega las clases de las métricas introducidas.

    Args:
        list_metrics (list): Lista de strings con los nombres de las métricas.

    Returns:
        list: Lista de las clases de las métricas.
    """
    metrics_mapping = get_metrics_mapping()
    return [metrics_mapping[met] for met in list_metrics]

def models_creator(dic_models):
    """
    Inicializa los modelos.

    Args:
        dic_models (dict): Diccionario de modelos entregado en el input.

    Returns:
        list: Lista de modelos instanciados.
    """
    models_mapping = get_models_mapping()
    # Lista vacía donde se guardará cada instancia de modelo.
    models = []
    for model_info in dic_models:

        # Lista de experimento individual.
        model_class = []

        # Obtenemos los parámetros de cada modelo.
        model_params = model_info.get('params', None)
        if model_params is not None:
            # Creamos la grilla de parámetros de modelo a probar.
            model_params_grid = list(ParameterGrid(model_info['params']))
        else:
            model_params_grid = [{}]

        # Obtenemos los preprocesamientos a probar por cada modelo.
        preprocesses = model_info.get('preprocesses', [])
        # Si no hay preprocesamientos, utilizamos el predeterminado.
        if len(preprocesses) == 0:
            for model_parameter_combination in model_params_grid:
                model_dic = {'params': model_parameter_combination}
                model = models_mapping[model_info['name']](**model_dic)
                model_class.append(model)

        # En otro caso, iteramos por cada preprocesamiento.
        else:
            for preprocess_info in preprocesses:

                # Diccionarios que contienen los kwargs de la instancia de
                # modelo y de preprocesamiento.
                model_dic = {}
                preprocess_dic = {}

                # Lista de instancias de tokenizadores.
                # Los tokenizers son fijos (se usan todos los de la lista).
                tokenizers = []
                tokenizers_info = preprocess_info.get('tokenizers', [])
                for tokenizer in tokenizers_info:
                    tokenizers.append(tokenizers_mapping[tokenizer]())
                preprocess_dic['tokenizers'] = tokenizers

                # Obtenemos los parámetros de cada preprocesamiento.
                preprocess_params = preprocess_info.get('params', None)
                if preprocess_params is not None:
                    # Creamos la grilla de parámetros de preprocesamiento
                    # a probar.
                    preprocess_params_grid = list(ParameterGrid(preprocess_params))
                else:
                    preprocess_params_grid = [{}]

                for model_parameter_combination in model_params_grid:
                    for preprocess_parameter_combination in preprocess_params_grid:
                        #Instanciamos el preprocess.
                        preprocess_dic['params'] = preprocess_parameter_combination
                        preprocess = preprocesses_mapping[preprocess_info['name']](preprocess_dic)

                        # Instanciamos el modelo.
                        model_dic['preprocess'] = preprocess
                        model_dic['params'] = model_parameter_combination
                        model = models_mapping[model_info['name']](**model_dic)

                        # Agregamos el modelo a la lista de modelos.
                        model_class.append(model)

        models.append(model_class)
    return models

def models_preprocess(model, datasets_list):
    """
    Aplica el preprocesamiento de model en cada uno de los datasets contenidos
    en la lista datasets_list.

    Args:
        model (Model): Modelo instanciado.
        datasets_list (list): Lista de datasets a preprocesar.

    Returns:
        list: Lista de datasets preprocesados.
    """
    return [model.preprocess(ds) for ds in datasets_list]

def get_set_result(models, metrics, X, y, exp_id):
    """
    Genera los resultados por metrica para una lista de modelos, dado un
    conjunto de entrenamiento (X e y), ademas guarda los resultados con un
    nombre unico.

    Args:
        models (list): Lista de listas que contienen los modelos.
        metrics (list): Lista que contiene las metricas que se usaran
            para generar el reporte.
        X (array): Arreglo con textos de testeo preprocesados.
        y (array): Matriz con las etiquetas reales de los textos
                        de testeo.
        exp_id: id del experimento del cual se esta llamando a esta funcion.

    Returns:
        dict: Diccionario con los resultados del modelo por metrica.
    """
    results = {}
    for j in range(len(models)):
        name = models[j][0].__class__.__name__
        for i in range(len(models[j])):
            X_prep, = models_preprocess(models[j][i], [X])
            y_pred = models[j][i].predict(X_prep)
            result = metrics_result(metrics, y, y_pred)
            results[f'{name}-{j}-{i}-{exp_id}'] = result
    return results


def models_tester(models, metrics, X_test, y_test, exp_id):
    """
    Realiza predicciones con respecto al set de testeo de los mejores modelos
    obtenidos en el trainer. Se genera el output del experimenter, al cual se
    le añaden los reportes de desempeño.

    Args:
        models (list): Lista de listas, en donde cada lista interna corresponde
                        a las n_best instancias de esa clase según el dataset de
                        entrenamiento.
        metrics (list): Lista con las clases de las métricas que se
                        introdujeron en show.
        X_test (array): Arreglo con textos de testeo preprocesados.
        y_test (array): Matriz con las etiquetas reales de los textos
                        de testeo.

    Returns:
        dict: Output del experimenter.
    """
    output = {}
    for j in range(len(models)):
        name = models[j][0].__class__.__name__
        output[f'{name} {j}'] = []
        for i in range(len(models[j])):
            X_te, = models_preprocess(models[j][i], [X_test])
            y_pred = models[j][i].predict(X_te)
            result_by_metric = metrics_result(metrics, y_test, y_pred)
            report = models_report(result_by_metric)
            model_dict = {
                'name': f'{name}-{j}-{i}-{exp_id}',
                'model_id': f'{datetime.now()}',
                'scores': report
            }
            output[f'{name} {j}'].append(model_dict)
    return output

def metrics_result(metrics, y_gold, y_pred):
    """
    Genera un reporte de desempeño de prediccción con respecto
    a las metricas especificadas. El reporte queda particionado
    por metrica.

    Args:
        metrics (list): Lista con las clases de las métricas especificadas.
        y_gold (array): Matriz con las labels reales.
        y_pred (array): Matriz con las labels predichas.

    Returns:
        dict: Reporte de desempeño
    """
    metrics_mapping = get_metrics_mapping()
    report = {}
    if len(metrics) == 0:
        metrics = metrics_mapping.values()
    for metric in metrics:
        result = metric(y_gold, y_pred)
        report[metric.name] = result
    return report

def models_report(results):
    """
    Genera reporte de resultados por label o nivel.

    Args:
        results (dict): Diccionartio con los resultados por metrica

    Returns:
        dict: Reporte de desempeño, particionado por label o nivel.
    """
    def union(metric_name, metric_value, report):
        for key in metric_value:
            if key in report:
                report[key][metric_name] = metric_value[key]
            else:
                report[key] = {}
                report[key][metric_name] = metric_value[key]
    report = {}
    metrics_results = results.copy()
    while metrics_results != {}:
        metric_list = metrics_results.popitem()
        union(metric_list[0],metric_list[1], report)
    return report

def models_trainer(models, X_train, X_val,
                    y_train, y_val):
    """
    Entrena los modelos y genera reportes en base a todas
    las métricas.

    Args:
        models (list): Lista de listas de modelos, en donde cada lista interna
                        corresponde a un modelo especificado en el input.
        X_train (array): Arreglo con textos de entrenamiento preprocesados.
        X_val (array): Arreglo con textos de validación preprocesados.
        y_train (array): Matriz con las etiquetas reales de los textos
                            de entrenamiento.
        y_val (array): Matriz con las etiquetas reales de los textos
                        de validación.

    Returns:
        list: Lista con los reportes de desempeño obtenidos con respecto
                al set de validación.
    """
    output = []
    for model_class in models:
        results = []
        for model in model_class:
            X_tr, X_va = models_preprocess(
                model, [X_train, X_val])
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_va)
            result_by_metric = metrics_result([], y_val, y_pred)
            report = models_report(result_by_metric)
            results.append({
                "model": model,
                "report": report
            })
        output.append(results)
    return output

def optimizer(reports, metrics_dic):
    """
    Genera un ranking de los mejores reportes con respecto a la etiqueta y
    métrica especificadas en el input. Luego, filtra los reportes de manera
    tal de dejar los n mejores (especificados por el usuario).

    Args:
        reports (list): Lista con los reportes obtenidos en el trainer.
        metrics_dic (dict): Diccionario con las métricas especificadas
                            por el usuario.

    Returns:
        list: Lista con los reportes filtrados.
    """
    opt_label, opt_metric, best_n = get_optimizer_params(metrics_dic)
    rank_reports(reports, opt_label, opt_metric)
    return filter_models(reports, best_n)

def rank_reports(unranked, label, metric):
    """
    Ordena una lista de listas de reportes con respecto a la label y métrica
    especificada.

    Args:
        unranked (list): Lista de listas de reportes.
        label (str): Nombre de la label con la cual se quiere optimizar.
        metric (str): Nombre de la métrica con la cual se quiere optimizar.
    """
    for report_list in unranked:
        report_list.sort(
            key=lambda dic: search_value(
                dic, label, metric))
        report_list.reverse()

def search_value(dic, first_key, second_key):
    """
    Retorna el valor obtenido de la métrica.

    Args:
        dic (dict): Diccionario que tiene un modelo y su reporte.
        first_key (str): String con la primera llave del reporte.
        second_key (str): String con la segunda llave del reporte.

    Returns:
        float: Valor obtenido.
    """
    return dic['report'][first_key][second_key]

def set_params_to_list(js):
    """
    Inserta en listas los parámetros ingresados individualmente.

    Args:
        js (dict): Diccionario con los parámetros modificados.
    """
    models = js['models']
    if isinstance(models, dict):
        js['models'] = [models]
    for model in models:
        model_params = model.get('params', {})
        for key in model_params:
            if not isinstance(model_params[key], list):
                model_params[key] = [model_params[key]]

        preprocesses = model.get('preprocesses', [])
        if not isinstance(preprocesses, list):
            model['preprocesses'] = [preprocesses]

        for preprocess in preprocesses:
            preprocess_params = preprocess.get('params', {})

            for key in preprocess_params:
                if not isinstance(preprocess_params[key], list):
                    preprocess_params[key] = [preprocess_params[key]]

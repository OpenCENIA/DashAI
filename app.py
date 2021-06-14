"""
API:
Contiene todo lo necesario para inciar el servicio web y manejar las distintas
consultas que se le hagan
"""
from flask import Flask, abort, jsonify, request, make_response, Response

import torch
import time

from worker import celery_app
from celery.result import AsyncResult
import os
from flask_sqlalchemy import SQLAlchemy
import json
import logging

# Inicializamos y configuramos la app de Flask
app = Flask(__name__)

app.config.from_object(os.environ.get("APP_SETTINGS", "config.DevelopmentConfig"))
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Enlazamos la app con una db mediada por SQLAlchemy (aqui es donde se guardara toda la informacion de los experimentos y modelos)
db = SQLAlchemy(app)

from models import Arquitectura, Metrica, Modelo, Experimento, Resultado, Dataset, Task, check_dataset_input
from experimenter import Experimenter, check_experimenter_input, set_params_to_list


# Creacion y configuracion del logger (servira para ver el estado de un proceso que se este realizando en la api)
logging.basicConfig(handlers=[logging.FileHandler(filename="logs.log", mode = 'w', encoding='utf-8')],
                    level = logging.DEBUG,
                    format= "%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

# Definicion del device correspondiente
if torch.cuda.is_available():
    dev = "cuda:0"
else:
   dev = "cpu"
device = torch.device(dev)

@app.route('/health')
def health():
    """
    Ruta que permite saber rapidamente si estan arriba los servicios asociados a la api.
    Recomendable consultar este endPoint cada vez que se enciende al api.
    """
    status = {}
    try:
        Modelo.query.get(1)
        db_status = "OK"
    except:
        db_status = "Error"
    status['db_status'] = db_status

    try:
        task = task_health.delay()
        time.sleep(2)
        if task.state == "SUCCESS":
            worker_status = "OK"
        else:
            worker_status = "BUSY"
    except:
        worker_status = "Error"
    status['worker_status'] = worker_status
    status['api_status'] = "OK"
    return make_response(status, 200)

@celery_app.task(name='app.task_health')
def task_health():
    """
    Metodo para probar el correcto funcionamiento del worker.
    """
    return 1+1

@app.route("/classifier", methods=['POST'])
def predict():
    """
    Ruta que permite dado un id de un modelo ya entrenado y una lista de textos,
    generar y entregar una prediccion sobre la o las etiquetas a dicha lista de textos.
    Esta prediciccion se genera usando el modelo asociado a la id asociada.

    Tanto el id del modelo como el texto se deben extraer de un diccionario ingresado al
    momento de realizar la request. El diccionario debe tener el siguiente formato:
    {
        'id' : int que represente el id del modelo que se usara para realizar la prediccion,
        'text': string o lista de string que contienen un texto al cual se le quiere calcular
        una prediccion de su(s) etiqueta(s) asociadas
    }
    """
    id_model = request.json.get('id')
    text = request.json.get("text")
    if text is None:
        abort(400)
    if id_model:
        model = Modelo.query.get(id_model)
        abort(404) if not model else None
    else:
        abort(400)

    if isinstance(text, str):
        text = [text]
    elif not hasattr(text, "__iter__"):
        abort(400)
    dic = {'text':text,'model_id':model.id}
    task = classifier_task.delay(dic)
    return make_response({'task_id':task.id},202)

@celery_app.task(name='app.classifier_task')
def classifier_task(dic):
    """
    Metodo que sera ejecutado por la ruta classifier, es el que dado el texto y
    el id de un modelo genera una prediccion (o una lista de ellas) y las entrega
    al usuario.
    Este metodo es usualmente llamado de la forma classifier_task.delay(dic) para poder
    ejecutarse fuera de la api.

    Args:
        dic (dict): Diccionario con la informacion necesaria para generar una prediccion.

    Returns:
        Retorna un diccionario que contiene la probabilidad de que cada sentencia tenga asociada
        una cierta etiqueta.
    """
    with app.app_context():
        text = dic.get('text')
        loaded_model = Modelo.query.get(dic['model_id']).load_model()
        texto_preprocesado = map(loaded_model.preprocess, [text])
        df = map(loaded_model.predict_proba, texto_preprocesado)
        result = map(json_parser, df)
        return jsonify(list(result))

@app.route("/experimenter", methods=['POST'])
def experimenter():
    """
    Ruta encargada de mandar a ejecutarse un experimento.

    Returns:
        dict: id (string) de la task de celery asociada al experimento.
    """
    logger.debug('Comenzando a experimentar')
    dic = request.json
    schema_result = check_experimenter_input(dic)
    abort(Response(schema_result.err_msg, status=400)) if not schema_result else None
    set_params_to_list(dic)
    task = experiment_task.delay(dic)
    return make_response({'task_id': task.task_id}, 202)

@celery_app.task(name='app.experiment_task')
def experiment_task(dic):
    """
    Funcion ejecutada por el worker encargada de realizar el experimento
    y guardar los modelos entrenados, los resultados de dichos modelos y cualquier
    otra informacion relevante del experimento.

    Args:
        dic (dict): Diccionario que contiene toda la informacion necesaria para poder
        inicializar un experimento.
    """
    with app.app_context():
        experiment = Experimenter(dic)
        experiment.create()
        experiment.train()
        experiment.optimize()
        experiment.test()
        experiment.save()
        return jsonify(experiment.output)

@app.route('/task/status/<id>')
def get_status(id):
    """
    Ruta que entrega el estado de task asociada al id ingresado.
    Estas pueden ser task asociadas a una clasificacion o a un experimento.

    Args:
        id (string): Codificacion asociada a una task de celery

    Returns:
        dict: Indicando todas las caracteristicas que permitan describir el estado
        actual del experimento
    """
    res = AsyncResult(id=id, app=celery_app)
    return {'status':res.status}

@app.route('/task/result/<id>')
def get_result(id):
    """
    Ruta que retorna el resultado de una task, sea esta exitosa (SUCCESS) o fallida (FAILURE),
    si no retorna una mensaje indicando que aun no esta lista la task.

    Args:
        id (string): Codificacion asociada a una task de celery

    Returns:
        En caso de que la task este lista, el resultado de la task, si no un mensaje indicando que
        aun no ha terminado.
    """
    res = AsyncResult(id=id,app=celery_app)
    if res.state == "SUCCESS" or res.state == "FAILURE":
        return res.get()
    else:
        return {'message':"The task isn't finished yet, please wait."}

@app.route("/createArq", methods=["POST"])
def create_arq():
    """
    Ruta para poder subir una Arquitectura a la base de datos del servidor.
    Para hacerlo se recomienda ejecutar el archivo request_arqs.py presente en el
    repositorio, en el se encuentran las instrucciones de su uso.

    Returns:
        Retorna un diccionario donde en la llave 'arq_id' se encuentra el id asociado
        a la nueva arquitectura creada.
    """
    name = request.json.get("name")
    data = request.json.get("data").encode()
    parametros = request.json.get("params")
    arq = Arquitectura(name, data, parametros)
    db.session.add(arq)
    db.session.commit()
    return make_response({'arq_id': arq.id},201)

@app.route("/deleteArq/<arq_id>", methods=['DELETE'])
def delete_arq(arq_id):
    """
    Ruta para poder eliminar una de las arquitecturas que ya hayan sido subidas
    a la aplicacion

    Args:
        arq_id (int): id de la arquitectura que se desea eliminar.

    Returns:
        Mensaje indicando el nombre de la arquitectura eliminada o que la arquitectura
        que se desea eliminar no existe.
    """
    arq = Arquitectura.query.get(int(arq_id))
    if arq:
        name = arq.name
        db.session.delete(arq)
        db.session.commit()
        return f"Arquitectura {name} eliminada"
    else:
        return "La arquitectura no existe"

@app.route('/updateArq', methods=["PATCH"])
def update_arq():
    """
    Ruta para poder actualizar el nombre o codigo de una arquitectura ya
    creada.
    Para hacerlo se recomienda ejecutar el archivo request_update_arq.py,
    en el se encuentran instrucciones de su uso.

    Returns:
        Retorna un mensaje indicando el exito de la actualizacion de la
        arquitectura.
    """
    id = request.json.get("id")
    name = request.json.get("name")
    data = request.json.get("data")
    if id:
        arq = Arquitectura.query.get(id)
        if arq:
            if name:
                arq.name = name
            if data:
                arq.data = data.encode()
            db.session.commit()
            return make_response(f"Arq {name} updated",201)
        else:
            abort(404)
    else:
        abort(400)

@app.route("/createModel", methods=["POST" ])
def create_model():
    """
    Ruta para poder subir modelos ya entrenados a la base de datos,
    esta pensando en que estos modelos sean usados como referencia para
    otros modelos al momento de testear nuevos modelos en un experimento.
    """
    data = request.files['json']
    json_data = json.load(data)
    name = json_data.get("name")
    arc_name = json_data.get("arc_name")
    params = json_data.get("params")
    instancia = request.files['file']
    arq = Arquitectura.query.filter_by(name=arc_name).first()
    model = Modelo(name, params, instancia.read())
    if model.validate_schema(arq):
        arq.models.append(model)
        db.session.commit()
        return make_response({'model': model.id}, 201)
    else:
        abort(400, 'Bad Schema')

@app.route("/createMetric", methods=["POST"])
def create_metric():
    """
    Ruta para poder crear una nueva metrica y subirla a la base de
    datos de la aplicacion.
    Para hacerlo se recomienda ejecutar el archivo request_metric.py,
    en el se encuentran instrucciones de su uso.

    Returns:
        Retorna un diccionario en el cual la llave 'metric_id' contiene
        el id asociado a la nueva metrica creada.
    """
    name = request.json.get("name")
    data = request.json.get("data").encode()
    metric = Metrica(name, data)
    db.session.add(metric)
    db.session.commit()
    return make_response({'metric_id': metric.id},201)

@app.route('/updateMetric', methods=["PATCH"])
def update_metric():
    """
    Ruta para poder actualizar el nombre o codigo de una metrica ya
    creada.
    Para hacerlo se recomienda ejecutar el archivo request_update_metric.py,
    en el se encuentran instrucciones de su uso.

    Returns:
        Retorna un mensaje indicando el exito de la actualizacion de la
        metrica.
    """
    id = request.json.get("id")
    name = request.json.get("name")
    data = request.json.get("data")
    if id:
        metric = Metrica.query.get(id)
        if metric:
            if name:
                metric.name = name
            if data:
                metric.data = data.encode()
            db.session.commit()
            return make_response(f"Metric {name} updated",201)
        else:
            abort(404)
    else:
        abort(400)


@app.route("/createDataset",methods=['POST'])
def create_dataset():
    result_schema = check_dataset_input(request.json)
    abort(Response(result_schema.err_msg, status=400)) if not result_schema else None
    json_doc = request.json.pop("dataset")
    name = request.json.pop("name")
    tags = request.json
    dataset = Dataset(json_doc, name, tags)
    verification = dataset.check_dataset()
    if verification:
        db.session.add(dataset)
        db.session.commit()
        return jsonify({"dataset_id": dataset.id, "name": dataset.name}), 201
    else:
        abort(Response(verification.err_msg, status=400))

@app.route('/models')
def get_model():
    """
    Ruta que entrega un diccionario con todos los modelos que se encuentran
    en la base de datos de la aplicacion. Estos aparecen junto con una serie
    de datos sobre ellos.
    """
    models = Modelo.query.all()
    return jsonify(list(map(lambda x: x.serialize(), models)))

@app.route('/arqs')
def get_arcs():
    """
    Ruta que entrega un diccionario con todas las arquitecturas que se encuentran
    en la base de datos de la aplicacion. Estas aparecen junto con una serie
    de datos sobre ellas.
    """
    arqs = Arquitectura.query.all()
    return jsonify(list(map(lambda x: x.serialize(), arqs)))

@app.route('/metrics')
def get_metrics():
    """
    Ruta que entrega un diccionario con todas las metricas que se encuentran
    en la base de datos de la aplicacion. Estas aparecen junto con una serie
    de datos sobre ellas.
    """
    mets = Metrica.query.all()
    return jsonify(list(map(lambda x: x.serialize(), mets)))

@app.route('/results')
def get_results():
    """
    Ruta que entrega un diccionario con todos los resultados que se encuentran
    en la base de datos de la aplicacion. Estos aparecen junto con una serie
    de datos sobre ellos.
    """
    res = Resultado.query.all()
    return jsonify(list(map(lambda x: x.serialize(), res)))

@app.route('/experiments')
def get_experiments():
    """
    Ruta que entrega un diccionario con todos los experimentos que se encuentran
    en la base de datos de la aplicacion. Estos aparecen junto con una serie
    de datos sobre ellos.
    """
    exps = Experimento.query.all()
    return jsonify(list(map(lambda x: x.serialize(), exps)))

@app.route('/datasets')
def get_datasets():
    """
    Ruta que entrega un diccionario con todos los datasets que se encuentran
    en la base de datos de la aplicacion. Estos aparecen junto con una serie
    de datos sobre ellos.
    """
    datasets = Dataset.query.all()
    return jsonify(list(map(lambda x: x.serialize(), datasets)))

@app.route('/tasks')
def get_tasks():
    """
    Ruta que entrega un diccionario con todas las taks que se encuentran
    en la base de datos de la aplicacion. Estas aparecen junto con una serie
    de datos sobre ellas.
    """
    df = Task.get_tasks()
    return df.to_html()

def json_parser(predicciones):
    """
    Metodo que permite modificar el output del metodo predict_proba de los modelos
    en un formato mas legible y entendible por una persona.

    Args:
        predicciones (df): Pandas DataFrame que contiene informacion sobre las predicciones
        que realizo un modelo sobre una lista de textos.

    Returns:
        Retorna un diccionario con la misma informacion que el input solo que mas legible por
        una persona.
    """
    output = {f"sentencia {idx}":[{
            "label": lab,
            "prob": predicciones[lab][idx]
            } for lab in predicciones.columns
            ] for idx in predicciones.index}
    return output

def json_multiparser(pred_dict):
    # Funcion que permite ordenar de mejor manera una lista de diccionarios con los resultados
    # de una prediccion. Actualmente no se encuentra en uso
    output = {mod:pred_dict[mod]  for mod in pred_dict}
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)

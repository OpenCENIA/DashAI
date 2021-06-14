"""
Modelos que capturan el modelo de la base de datos
"""
from app import db
from jsonschema import validate, ValidationError
from datetime import datetime
import enum
from hashlib import sha512
import base64
import os
from io import StringIO

import psycopg2
from config import Config
import pandas as pd

DATA_PATH = 'db/'
MODELOS_PATH = os.path.join(DATA_PATH, 'modelos')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(MODELOS_PATH):
    os.makedirs(MODELOS_PATH)

# Tablas intermedias entre entidades
modelos_experimentos = db.Table('modelos_experimentos',
                                db.Column('model_id', db.Integer, db.ForeignKey('modelos.id'), primary_key=True),
                                db.Column('experiment_id', db.Integer, db.ForeignKey('experimentos.id'), primary_key=True),
                                db.Column('hiperparametros', db.JSON)
                                )

experimentos_metrics = db.Table('experimentos_metrics',
                                db.Column('experiment_id', db.Integer, db.ForeignKey('experimentos.id'), primary_key=True),
                                db.Column('metric_id', db.Integer, db.ForeignKey('metricas.id'), primary_key=True))

# Funciones auxiliares
def arquitectures_mapping():
    arqs = Arquitectura.query.all()
    return {
        key : value for key, value in zip(map(lambda x: x.name, arqs), map(lambda x: x.get_class(), arqs))
    }

def metrics_mapping():
    mets = Metrica.query.all()
    return {
        key : value for key, value in zip(map(lambda x: x.name, mets), map(lambda x: x.get_class(), mets))
    }

def get_class_from_bytes(data, export_key):
        exec(data.decode(), globals(), globals())
        clase = globals().get(export_key)
        return clase


dataset_schema = {
    'type' : 'object',
    'properties': {
        'name' : {
            'type' : 'string',
            'minLength': 6,
            'maxLength': 60,
            'error_msg': 'Largo debe ser entre 6 y 60 caracteres'
        },
        'dataset' : {
            'type' : 'object',
            'required': ['response'],
            'properties': {
                'response': {"error_msg": '"dataset" debe tener el campo "response"',
                    'type': 'object',
                    'required': ['docs'],
                    'properties':{
                        'docs': {
                            'error_msg': '"response" de "dataset" debe tener campo "docs"',
                            'type': 'array',
                            'items': {
                                'type': 'object'
                            }
                        }
                    }
                },
            }
            },
        'task': {'type' : 'string',
                 'enum': ['binary', 'multiclass', 'multilabel'],
                 'error_msg': 'Task debe ser "binary", "multiclass" o "multilabel"'
                  },
        'x': {'type': 'array', 'items': {'type': 'string', 'error_msg': 'Todas entradas de "x" deben ser string'},
              'error_msg': '"x" debe ser un arreglo de strings con columnas del csv'},
        'y': {'type': 'string', 'error_msg': '"y" debe ser un string para una columna del csv, que indica cuales serán los labels'},
        'split_sizes': {'type': 'array',
                        'items': {
                            'type': 'number',
                            'exclusiveMinimum': 0,
                            'exclusiveMaximum': 1,
                            'error_msg': '"split_sizes" debe tener 3 números que sumen 1'
                            },
                        'minItems': 3,
                        'maxItems': 3,
                        'error_msg': '"split_sizes", debe tener exactamente 3 números que sumen 1, y estar en el intervalo (0,1)'
                        },
    },
    'required': ['dataset', 'x', 'y', 'name'],
    'additionalProperties': False,
    'error_msg': 'Se requieren los campos "dataset", "x", "y" y "name", y opcionalmente los campos "task" y "split_sizes"'
}

class VerificationResult:

    def __init__(self, accepted: bool, err_msg=""):
        self.accepted = accepted
        self.err_msg = err_msg

    def __bool__(self):
        return self.accepted

def check_dataset_input(data):
    try:
        validate(data, dataset_schema)
    except ValidationError as err:
        return VerificationResult(False, err.schema["error_msg"])
    if 'split_sizes' in data:
        suma = sum(data["split_sizes"])
        valid_split =  1 - 1e-5 <= suma <= 1 + 1e-5
        err_msg = "Los valores de 'split_sizes' deben sumar 1"
        return VerificationResult(True) if valid_split else VerificationResult(False, err_msg)
    else:
        return VerificationResult(True)

# Entidades principales
class ResultKind(enum.Enum):
    # Tipo de resultado, para evitar ambigüedades

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class TaskStatus(enum.Enum):
    # Estatus de tarea, para evitar ambigüedades

    PENDING = "pending"
    DONE = "done"
    ERROR = "error"
    RUNNING = "running"

class Arquitectura(db.Model):

    __tablename__ = 'arquitecturas'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), unique=True)
    data = db.Column(db.LargeBinary, nullable=False)
    models = db.relationship("Modelo", backref="arquitectura", lazy=True)
    params_schema = db.Column(db.JSON, nullable=False)
    class_name = db.Column(db.String(20), nullable=False)

    def __init__(self, name, data, schema):
        """
        Inicializa la fila que se guardará en la BD

        name (Str): Nombre de la arquitectura
        data (LargeBinary): Binario que contiene el código de la arquitectura señalada
        schema (JSON): Archivo formato json-schema para verificar el input de la API
        """
        self.name = name
        self.data = data
        self.params_schema = schema
        self.class_name = self.get_class().__name__


    def get_class(self):
        """
        Devuelve el nombre de clase de la arquitectura
        """
        clase = get_class_from_bytes(self.data,"MODEL")
        clase.schema = self.params_schema
        return clase

    def serialize(self):
        """
        Devuelve una serialización de la arquitectura
        """
        clase = self.get_class().__name__
        return {
            'id': self.id,
            'name': self.name,
            'clase': clase,
            'schema': self.params_schema
        }

class Modelo(db.Model):

    __tablename__ = 'modelos'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    params = db.Column(db.JSON, nullable=False)
    path = db.Column(db.String(120), nullable=False)
    arc = db.Column(db.Integer, db.ForeignKey('arquitecturas.id'), nullable=False)
    experiments = db.relationship("Experimento", secondary=modelos_experimentos,
                                    back_populates="models")
    results = db.relationship("Resultado", backref="modelo", lazy=True)

    def __init__(self, name, params, instancia, arc_id):
        """
        Inicializa la fila que se guardará en la BD

        name (Str): Nombre del modelo
        params (JSON): Archivo formato json-schema para verificar el input de la API
        instancia (LargeBinary): Binario que contiene el código de la arquitectura señalada
        arc_id (Int): Id de la arquitectura que se usó para crear el modelo
        """
        self.name = name
        self.params = params
        hash_data = sha512(instancia)
        digested = hash_data.digest()
        self.path = base64.b32encode(digested).decode()
        with open(os.path.join(MODELOS_PATH, self.path +'.txt'), 'wb') as file:
            file.write(instancia)
        self.arc = arc_id

    def delete_file(self):
        try:
            os.remove(os.path.join(MODELOS_PATH, self.path + '.txt'))
        except FileNotFoundError:
            pass

    def validate_schema(self, arq=None):
        """
        Returna si el schema es, o no, válido
        """
        if arq is None:
            arq = self.arquitectura
        try:
            validate(self.params, arq.params_schema)
            return True
        except ValidationError:
            return False

    def load_model(self):
        """
        Función que carga la arquitectura desde el path interno
        """
        arq = self.arquitectura
        clase = arq.get_class()
        return clase.load(os.path.join(MODELOS_PATH, self.path + ".txt"))

    def serialize(self):
        """
        Devuelve una serialización del modelo
        """
        return {
            'id': self.id,
            'name': self.name,
            'params': self.params,
            'arc_id': self.arc,
            'path': self.path
        }

class Experimento(db.Model):

    __tablename__ = 'experimentos'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False)
    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)
    status = db.Column(db.Enum(TaskStatus), default=TaskStatus.PENDING)
    log = db.Column(db.Text)
    models = db.relationship("Modelo", secondary=modelos_experimentos,
                                back_populates="experiments")
    results = db.relationship("Resultado", backref="experiment", lazy=True)
    metrics = db.relationship("Metrica", secondary=experimentos_metrics,
                                backref="experiments")


    def __init__(self, log_text=None):
        """
        Inicializa la fila que se guardará en la BD

        log_text (Str): Logs correspondientes a la ejecución de este experimento.
        """
        self.created_at = datetime.now()
        self.log = log_text
        self.iter_status = self.generate_status()
        self.status = next(self.iter_status)

    def generate_status(self):
        """
        Se retorna el status en que se encuentre la tarea
        """
        lista = [
            TaskStatus.PENDING,
                TaskStatus.RUNNING,
                TaskStatus.DONE
                ]
        for i in range(len(lista)):
            yield lista[i]

    def serialize(self):
        """
        Devuelve una serialización del experimento
        """
        return {
            'id': self.id,
            'created_at': self.created_at,
        }

class Metrica(db.Model):

    __tablename__ = 'metricas'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), unique=True)
    data = db.Column(db.LargeBinary, nullable=False)
    results = db.relationship("Resultado", backref="metric", lazy=True)

    def __init__(self, name, data):
        """
        Inicializa el modelo que se guardará en la BD

        name (Str): Nombre de la métrica
        data (LargeBinary): Binario que contiene el código de la métrica señalada
        """
        self.name = name
        self.data = data

    def get_class(self):
        """
        Devuelve la clase de la métrica
        """
        return get_class_from_bytes(self.data,"METRIC")

    def serialize(self):
        """
        Devuelve una serialización de la métrica
        """
        return {
            'id': self.id,
            'name': self.name
        }

class Resultado(db.Model):

    __tablename__ = 'resultados'
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.Enum(ResultKind))
    value = db.Column(db.Float)
    model_id = db.Column(db.Integer, db.ForeignKey('modelos.id'), nullable=False)
    metric_id = db.Column(db.Integer, db.ForeignKey('metricas.id'), nullable=False)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experimentos.id"), nullable=False)

    def __init__(self, kind, value, metric_id):
        """
        Inicializa el modelo que se guardará en la BD

        name (Str): Nombre de la arquitectura
        data (LargeBinary): binario que contiene el código de la arquitectura señalada
        schema (JSON): Archivo formato json-schema para verificar el input de la API
        """
        assert self.check_result_kind(kind)
        self.kind = kind
        self.value = value
        self.metric_id = metric_id

    def serialize(self):
        """
        Devuelve una serialización del resultado
        """
        return {
            'id': self.id,
            'kind': self.kind.name,
            'metric_id': self.metric_id,
            'model_id': self.model_id,
            'exp_id': self.experiment_id
        }

    @staticmethod
    def check_result_kind(value):
        """ Método que valida los tipos (kind) de resultados"""
        return value in map(lambda x: x.name, ResultKind)

class Dataset(db.Model):

    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), nullable=False, unique=True)
    train = db.Column(db.PickleType, nullable=False)
    val = db.Column(db.PickleType, nullable=False)
    test = db.Column(db.PickleType, nullable=False)
    tags = db.Column(db.JSON, nullable=False)

    def __init__(self, json_doc, name, tags):
        """
        Inicializa la fila que se guardará en la BD

        json_doc (JSON): El archivo json con la data que se quiere guardar
        name (String): Nombre con el que se identificara el Dataset
        tags (JSON): Diccionario con toda la informacion asociada al dataset,
                    por ejemplo, entrada(s), etiqueta(s), la task asociada al Dataset, etc.
        """
        self.data = json_doc
        self.name = name
        self.tags = tags
        self.tags.setdefault('split_sizes', [0.7, 0.15, 0.15])
        self.tags.setdefault('task', 'multilabel')

    def check_dataset(self):
        try:
            df = pd.DataFrame(self.data['response']['docs'])
        except Exception:
            return VerificationResult(False, '"dataset" contiene documentos invalidos o inconsistentes en dataset["response"]["docs"]')
        if not pd.Series(self.tags["x"]).isin(df.columns).all():
            return VerificationResult(False, 'Hay una columna en "x" que no está en "dataset"')
        if not self.tags["y"] in df.columns:
            return VerificationResult(False, 'La columna indicada por "y", no está en "dataset"')
        task_verification = self.check_task(df)
        if not task_verification:
            return task_verification

        self.df = df
        self.split()
        return VerificationResult(True)

    def transform(self):
        x, y = self.tags["x"], self.tags["y"]
        labels = self.df[y].str.split("|")
        frame = pd.Series(list(set(labels.sum())), name='labels').to_frame()
        for i in range(labels.shape[0]):
            frame[i] = frame['labels'].isin(labels.iloc[i]).apply(int)

        frame = frame.set_index("labels").T
        columnas = frame.sum().sort_values(ascending=True).index
        frame = frame[columnas]
        aux = self.df[x]
        aux[columnas] = frame
        self.df = aux

    def split(self):
        self.transform()
        split_sizes = self.tags["split_sizes"]
        val_fraction = split_sizes[1]/sum(split_sizes[1:])
        self.train = self.df.sample(frac=split_sizes[0])
        self.val = self.df.drop(index=self.train.index).sample(frac=val_fraction)
        self.test = self.df.drop(index=self.train.index.union(self.val.index))

    def check_task(self, df):
        y = self.tags["y"]
        task = self.tags["task"]
        multilabel = (df[y].str.split("|").apply(len) > 1).any()
        multiclass = len(df[y].unique()) > 2 and not multilabel
        binary = len(df[y].unique()) == 2
        if not (multilabel or multiclass or binary):
            return VerificationResult(False, f'La columna "{y}" no cumple ningún formato aceptado')
        falla = False
        if multilabel and not task == 'multilabel':
            falla = True
        elif multiclass and not task == 'multiclass':
            falla = True
        elif binary and not task == 'binary':
            falla = True
        if falla:
            return VerificationResult(False, f'La columna "{y}" no cumple el formato de {task}')
        return VerificationResult(True)
    
    def serialize(self):
        """
        Devuelve una serialización del dataset
        """
        return {
            'id': self.id,
            'name': self.name,
            'tags': self.tags
        }




class Schema(db.Model):

    __tablename__ = 'schemas'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)
    schema = db.Column(db.JSON, nullable=False)

class Task:

    CONNECTION = None

    @classmethod
    def connect(cls):
        DB_URI = Config.SQLALCHEMY_DATABASE_URI
        cls.CONNECTION = psycopg2.connect(DB_URI)

    @classmethod
    def get_tasks(cls, df=True):
        if cls.CONNECTION is None:
            cls.connect()
        cursor = cls.CONNECTION.cursor()
        query = """SELECT task_id, status FROM celery_taskmeta;"""
        cursor.execute(query)
        if df:
            return pd.DataFrame(cursor.fetchall(), columns=["Task_id", "Status"])
        return cursor

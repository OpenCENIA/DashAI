import json

from io import BytesIO
import joblib
#import pandas as pd
#from Models.preprocess.distil_emb import DistilBertEmbedding
#from skmultilearn.adapt import MLkNN
from sklearn.neighbors import KNeighborsClassifier

from Models.classes.model import Model

#version == 1.0.4
class KNN(Model, KNeighborsClassifier):
    """
    K Nearest Neighbors es un método de clasificación supervizado, 
    que determina la probabildiad de que un elemento pertenezca a una determinada 
    clase, considerando a sus k vecinos más cercanos.
    """
    MODEL = "KNN"

    # Task vars
    TASK = ["TEXT"]
    LABEL = "SINGLE"
    INSTANCE = "SINGLE"

    with open('Models/parameters/models_schemas/knn.json') as f:
        schema = json.load(f)
    
    def save(self, filename=None):

        if filename is None:
            bytes_container = BytesIO()
            joblib.dump(self, bytes_container)
            bytes_container.seek(0)
            return bytes_container.read()
        else:
            joblib.dump(self, filename)

    @staticmethod
    def load(filename):

        model = joblib.load(filename)
        return model

    # def __init__(self, **kwargs):
    #     """
    #     Inicializa una instancia de MLKNN.

    #     Kwargs:
    #         preprocess (PreProcess): Preprocesamiento instanciado, que contiene
    #                                     parámetros y tokenizadores inicializados.
    #         params (dic):   diccionario que contiene los hiperparámetros que
    #                         utiliza el modelo.

    #     params:
    #         k (int): Número de vecinos que se consideran en cada input para la clasificación. 
    #                     Default: 10.
    #         s (float): Parámetro de suavizamiento. Default: 1.
    #         ignore_first_neighbours (int): Permite ignorar los primeros vecinos. Default: 0.
    #     """
    #     preprocess = kwargs.get('preprocess', None)
    #     if preprocess is None:
    #         preprocess = DistilBertEmbedding(
    #             {'params': {}, 'tokenizer': []})
    #     super().__init__(preprocess)

    #     self.params = kwargs.get('params', {})
    #     self.params['k'] = self.params.get('k', 10)
    #     self.params['s'] = self.params.get('s', 1)
    #     self.params['ignore_first_neighbours'] = self.params.get('ignore_first_neighbours', 0)
    #     self.ml_knn = MLkNN(**self.params)

    # def fit(self, x, y=None):
    #     """
    #     Metodo que se usa para entrenar el modelo, no debe retornar nada.

    #     x (array-like): Arreglo donde cada componente tiene un texto ya preprocesado por el preprocess.
    #     y (array-like): Arreglo donde cada componente tiene un arreglo binario indicando si se prensenta o no la label
    #     """
    #     self.ml_knn.fit(x, y)

    # def predict(self, x):
    #     """
    #     Método que se usa para preprocesar los datos y de esta forma generar el
    #     input que recibe el modelo en fit, predict y predict_proba.
        
    #     x (array-like): Arreglo donde cada componente tiene un texto plano que necesita ser preprocesado.
        
    #     Retorna arreglo con etiquetas multilabel, pd.DataFrame
    #     """
    #     return self.ml_knn.predict(x).toarray()

    # def predict_proba(self, x):
    #     """
    #     x (array-like): Arreglo donde cada componente tiene un texto ya preprocesado por el preprocess.
    #     Retorna arreglo de probabilidad multilabel, pd.DataFrame
    #     """
    #     if x.ndim == 1:
    #         x = x.reshape(1, -1)
    #     return pd.DataFrame(self.ml_knn.predict_proba(x).toarray())
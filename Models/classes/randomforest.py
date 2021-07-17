import json
import logging

from io import BytesIO
import joblib
import numpy as np
import pandas as pd
from Models.preprocess.distil_emb import DistilBertEmbedding
from sklearn.ensemble import RandomForestClassifier

from Models.classes.model import Model

logger = logging.getLogger()

#versión == 1.0.4
class RandomForest(Model):
    """
    Random Forest de multi arboles de decisión. Especializada para
    múltiples etiquetas.

    Model: Clase padre de todos los modelos implementados.
    """
    MODEL = "RandomForest"
    with open('Models/parameters/models_schemas/randomforest.json') as f:
        schema = json.load(f)

    def __init__(self, **kwargs):
        """
        Inicializa una instancia de RandomForest.

        Kwargs:

            preprocess (PreProcess): preprocesamiento instanciado, que contiene
                                        parámetros y tokenizadores inicializados.

            params (dic):   diccionario que contiene los hiperparámetros que
                            utiliza el modelo.


        params:
            n_estimators (int): Número de árboles de decisión. Default: 200.
            depth (int): Profundidad máxima del árbol. Default: 30.
            min_samples (int): El mínimo número de samples requeridos para
                                particionar un nodo interno. Default: 2.
            min_leaf (int): El mínimo número de samples requeridas para estar
                            en un nodo hoja. Default: 1.
            node_terminal (int): máximo número de nodos hojas. Default: None.
            X_val (array-like): Arreglo de sentencias preprocesadas utilizadas
                                como conjunto de validación.
            y_val (array-like): Arreglo con entradas binarias, indicando si las
                                etiquetas que pertenecen a cada sentencia.
        """
        preprocess = kwargs.get('preprocess', None)
        if preprocess is None:
            preprocess = DistilBertEmbedding(
                {'params': {}, 'tokenizer': []})
        super().__init__(preprocess)

        self.params = kwargs.get('params', {})
        self.params['random_state'] = 0
        self.params['n_estimators'] = self.params.get("n_estimators", 200)
        self.params['max_depth'] = self.params.get("max_depth", 30)
        self.params['min_samples_split'] = self.params.get(
            "min_samples_split", 2)
        self.params['min_samples_leaf'] = self.params.get(
            "min_samples_leaf", 1)
        self.params['max_leaf_nodes'] = self.params.get(
            "max_leaf_nodes", None)
        
        self.kwargs = kwargs
        self.rf = RandomForestClassifier(**self.params)

    def fit(self, x, y=None):
        """
        Metodo que se usa para entrenar el modelo, no debe retornar nada.

        x (array-like): Arreglo donde cada componente tiene un texto ya preprocesado por el preprocess.
        y (array-like): Arreglo donde cada componente tiene un arreglo binario indicando si se prensenta o no la label
        """
        self.rf.fit(x, y)

    def predict(self, x):
        """
        Metodo que se usa para preprocesar los datos y de esta forma generar el
        input que recibe el modelo en fit, predict y predict_proba.

        x (array-like): Arreglo donde cada componente tiene un texto plano que necesita ser preprocesado.
        Retorna arreglo con etiquetas multilabel, pd.DataFrame
        """
        return self.rf.predict(x)

    def predict_proba(self, x):
        """
        x (array-like): Arreglo donde cada componente tiene un texto ya preprocesado por el preprocess.

        Retorna arregla de probabilidad multilabel, pd.DataFrame
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return pd.DataFrame(np.transpose(np.array(self.rf.predict_proba(x))[:,:,1]))

    def save(self, filename=None):
        """
        Módulo encargado de guardar la instancia entrenada del modelo

        filename (String): Nombre/path de destino donde se guardará el modelo entrenado
        """
        if filename is None:
            bytes_container = BytesIO()
            joblib.dump(self, bytes_container)
            bytes_container.seek(0)
            return bytes_container.read()
        else:
            joblib.dump(self, filename)


    @staticmethod
    def load(filename):
        """
        Módulo encargado de cargar un modelo ya entrenado con esta arquitectura
        
        filename (String): Nombre/path de destino de donde se obtendrá el modelo entrenado
        
        Retorna modelo intanciado de la clase RandomForest
        """
        model = joblib.load(filename)
        return model


MODEL = RandomForest

if __name__ == "__main__":
    pass

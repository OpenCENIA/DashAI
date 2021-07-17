import json
from io import BytesIO
import joblib
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

from Models.preprocess.tfidf import TFIDFPreprocess
from Models.classes.model import Model

#version == 1.0.4
class AdaBoost(Model):
    """
    Adaptive Boosting.

    Model: Clase padre de todos los modelos implementados.
    """
    MODEL = "AdaBoost"
    with open('Models/parameters/models_schemas/adaboost.json') as f:
        schema = json.load(f)
    
    def __init__(self, **kwargs):
        """
        Inicializa una instancia de AdaBoost

        Kwargs:
            preprocess (Preprocess): preprocesamiento instanciado, que contiene
                                        parámetros y tokenizadores inicializados.
            params (dic): diccionario que contiene los hiperparámetros que
                            utiliza el modelo.
        """
        preprocess = kwargs.get('preprocess', None)
        if preprocess is None:
            preprocess = TFIDFPreprocess(
                {'params': {}, 'tokenizer': []})
        super().__init__(preprocess)

        self.params = kwargs.get('params', {})
        self.params['n_estimators'] = kwargs.get('n_estimators', 100)
        self.params['learning_rate'] = kwargs.get('learning_rate', 0.1)
        self.params['random_state'] = kwargs.get('random_state', 0)
        self.kwargs = kwargs
        self.model = OneVsRestClassifier(AdaBoostClassifier(**self.params))

    def fit(self, x, y=None):
        '''
        x (array-like): Vectores que representan a las sentencias
                        tras transformarlas por un TfidfVectorizer, es decir, luego de usar
                        TfidfVectorizer.fit_transform sobre el corpus de documentos de train.
        y (array-like): Labels (o multilabels)
        '''
        self.model.fit(x, y)

    def predict(self, x):
        '''
        x (array-like): Vectores que representan a las sentencias tras transformarlas
                        con un TfidfVectorizer, es decir, luego de usar 
                        TfidfVectorizer.transform sobre el corpus de documentos sobre el que
                        se quiere predecir.

        Retorna una lista de listas con las labels que se le asignan
        a cada sentencia (1 si la tiene, 0 si no)
        '''
        return self.model.predict(x)

    def predict_proba(self, x):
        '''
        x (array-like): Vectores que representan a las sentencias
                        tras transformarlas con un TfidfVectorizer, es decir, luego de usar
                        TfidfVectorizer.transform sobre el corpus de documentos sobre el que
                        se quiere predecir.
        '''
        proba = self.model.predict_proba(x)
        return pd.DataFrame(proba)

    def save(self, filename=None):
        """
        filename (String): Nombre/path de destino donde se guardará el modelo entrenado. 
                        Si no es dado, se retorna un archivo. 
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
        """
        model = joblib.load(filename)
        return model

MODEL = AdaBoost

if __name__ == '__main__':
    pass

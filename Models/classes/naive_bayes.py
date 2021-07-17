import json
import logging

from io import BytesIO
import joblib
import pandas as pd
from Models.preprocess.bow import BOWPreprocess
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from Models.classes.model import Model

logger = logging.getLogger()

#version == 1.0.4
class NaiveBayes(Model):
    """
    Naive Bayes.

    Model: Clase padre de todos los modelos implementados.
    """
    MODEL = "NaiveBayes"
    with open('Models/parameters/models_schemas/naive_bayes.json') as f:
        schema = json.load(f)
    
    def __init__(self, **kwargs):
        """
        Crea una instancia del modelo probabilístico multinomial Naive Bayes.
        Kwargs:
            preprocess (PreProcess): Objeto encargado de preprocesar el texto
                                    entrante para que el modelo pueda tratarlo.
                                    Se sugiere dejar este atributo por defecto como Bag of Words.
            params (dic): diccionario que contiene los hiperparámetros que
                            utiliza el modelo.
            n_jobs (Integer): cantidad de hilos en paralelo utilizados por el modelo.
            
        params (dict):
            alpha (Float [0,1]): Coeficiente de suavidad del modelo.

        """
        preprocess = kwargs.get('preprocess', None)
        if preprocess is None:
            preprocess = BOWPreprocess({'params':{}, 'tokenizer': []})
        super().__init__(preprocess)

        self.params = kwargs.get('params', {})

        self.kwargs = kwargs
        self.model = OneVsRestClassifier(
            MultinomialNB(**self.params), n_jobs=-1)

    def fit(self, x, y):
        """
        x (array-like): Vectores que representan a las sentencias
                        tras transformarlas por un CountVectorizer, es decir, luego de usar
                        CountVectorizer.fit_transform sobre el corpus de documentos de train.
        y (array-like): Arreglo donde cada componente tiene un arreglo binario 
                        indicando si se presenta, o no, la label
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        x (array-like): Vectores que representan a las sentencias
                        tras transformarlas con un CountVectorizer, es decir, luego de usar
                        CountVectorizer.transform sobre el corpus de documentos sobre el que
                        se quiere predecir.

        Output: una lista de listas con las labels que se le asignan
                a cada sentencia (1 si la tiene, 0 si no)
        """
        if x.ndim == 1:
            x = x.reshape(-1,1)
        return self.model.predict(x)

    def predict_proba(self, x):
        """
        x (array-like): Vectores que representan a las sentencias
                        tras transformarlas con un CountVectorizer, es decir, luego de usar
                        CountVectorizer.transform sobre el corpus de documentos sobre el que
                        se quiere predecir.
        """
        proba = self.model.predict_proba(x)
        return pd.DataFrame(proba)

    def save(self, filename=None):
        """
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
        filename (String): Nombre/path de destino de donde se obtendrá el modelo entrenado

        Retorna modelo instanciado de la clase NaiveBayes
        """
        model = joblib.load(filename)
        return model

MODEL = NaiveBayes

if __name__ == '__main__':
    # Hacer caso de prueba
    pass

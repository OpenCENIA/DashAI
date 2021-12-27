import json
import logging

from io import BytesIO
import joblib

#import pandas as pd
#from Models.preprocess.distil_emb import DistilBertEmbedding
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from Models.classes.model import Model

logger = logging.getLogger()

#version == 1.0.4
class SVM(Model, KNeighborsClassifier):
    """
    Support vector machine. Algoritmo de aprendizaje supervisado que 
    separa dos clases en dos espacios mediante un hiperplano. Este hiperplano
    es definido como un vector llamado vector de soporte. Para poder implementar
    la funcionalidad de multietiqueta, se utiliza OneVsRestClassifier(SVC(**self.params))
    que realiza tantos vectores de soporte como etiquetas.

    Model: Clase padre de todos los modelos implementados.
    """
    MODEL = "knn"

    # Task vars
    TASK = ["TEXT"]
    LABEL = "SINGLE"
    INSTANCE = "SINGLE"

    with open('Models/parameters/models_schemas/svm.json') as f:
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
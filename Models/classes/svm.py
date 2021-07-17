import json
import logging

from io import BytesIO
import joblib
import pandas as pd
from Models.preprocess.distil_emb import DistilBertEmbedding
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from Models.classes.model import Model

logger = logging.getLogger()

#version == 1.0.4
class SVM(Model):
    """
    Support vector machine. Algoritmo de aprendizaje supervisado que 
    separa dos clases en dos espacios mediante un hiperplano. Este hiperplano
    es definido como un vector llamado vector de soporte. Para poder implementar
    la funcionalidad de multietiqueta, se utiliza OneVsRestClassifier(SVC(**self.params))
    que realiza tantos vectores de soporte como etiquetas.

    Model: Clase padre de todos los modelos implementados.
    """
    MODEL = "SVM"
    with open('Models/parameters/models_schemas/svm.json') as f:
        schema = json.load(f)

    def __init__(self, **kwargs):

        """
        Kwargs:
            preprocess (Preprocess): preprocesamiento instanciado, que contiene
                                        parámetros y tokenizadores inicializados.
            params (dic): diccionario que contiene los hiperparámetros que
                            utiliza el modelo.

        params (dict):
            probability (bool): True si se quiere predecir por probabilidades. Default: True.
            kernel (str): linear, poly, rbf, sigmoid. Es el kernel a utilizar en el modelo. Default=rbf
            gamma (str o float): scale, auto} o float, default=’scale’. Coeficiente para los kernels rbf, poly y sigmoid.
            coef0 (float): Default=0.0. Valor independiente del kernel. Solo es significante para kernel poly y sigmoid.
        """

        preprocess = kwargs.get('preprocess', None)
        if preprocess is None:
            preprocess = DistilBertEmbedding(
                {'params': {}, 'tokenizer': []})
        super().__init__(preprocess)

        self.params = kwargs.get('params', {})
        self.params['probability'] = self.params.get('probability', True)
        self.params['kernel'] = self.params.get('kernel', 'rbf')
        self.params['gamma'] = self.params.get('gamma', 0.1)
        
        self.kwargs = kwargs
        self.ml_svm = OneVsRestClassifier(SVC(**self.params))

    def fit(self, x, y=None):
        """
        Método que se usa para entrenar el modelo, no debe retornar nada.

        x (array-like): Arreglo donde cada componente tiene un texto ya preprocesado por el preprocess.
        y (array-like): Arreglo donde cada componente tiene un arreglo binario indicando si se prensenta o no la label
        """

        self.ml_svm.fit(x, y)

    def predict(self, x):
        """
        Método que se usa para predecir.

        x (array-like): Arreglo donde cada componente tiene un texto ya
                            preprocesado por preprocess.

        Retorna matriz de adyacencia, que indica la pertenencia de
        de las labels a las sentencias. En las filas se
        representan las sentencias (en orden), y en las
        columnas las etiquetas (en orden).
        """
        return self.ml_svm.predict(x)

    def predict_proba(self, X):
        """
        Este método es posible de realizar cuando 'probability = True'.
        Entrega la probabilidad de que cada etiqueta pertenezca a las
        distintas sentencias.

        X (array-like): Arreglo donde cada componente tiene un texto ya
                        preprocesado por preprocess.

        Retorna dataFrame, donde cada sentencia es representada
        en una fila, y en cada columna se representa una
        etiqueta. En la entrada (m,n) se puede observar la
        probabilidad de que en la m-ésima sentencia esté
        la n-ésima etiqueta.
        """
        aux = pd.DataFrame(self.ml_svm.predict_proba(X))
        return aux

    def save(self, filename=None):

        """
        Método encargado de guardar la instancia entrenada del modelo.

        filename (str, optional): Nombre/path de destino donde se guardará
                                    el modelo entrenado. Default: None.
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
        Método encargado de cargar un modelo ya entrenado con esta
        arquitectura.

        filename (str): Nombre/path de destino desde donde
                        se obtendrá el modelo entrenado.
                        Default: None.

        Retorna modelo previamente guardado.
        """

        model = joblib.load(filename)
        return model

MODEL = SVM

if __name__ == "__main__":
    pass
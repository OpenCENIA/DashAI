import json
import logging

from io import BytesIO
import joblib
import pandas as pd
from Models.preprocess.bow import BOWPreprocess
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from Models.classes.model import Model

logger = logging.getLogger()

#version == 1.0.4
class MLLogisticRegression(Model):
    """
    Modelo de regresión logística, con generalización multi-label.

    Model: Clase padre de todos los modelos implementados.
    """
    MODEL = "ML_LR"
    with open('Models/parameters/models_schemas/ml_lr.json') as f:
        schema = json.load(f)

    def __init__(self, **kwargs):
        """
        Inicializa el modelo MLLogisticRegression. 
        
        El modelo tiene dos técnicas: One vs Rest, que entrena una regresión 
        logística para cada etiqueta; y Classifier Chains, que entrena sucesivamente
        el modelo para cada etiqueta, intentando capturar correlaciones
        entre diversas etiquetas.

        Kwargs:
            preprocess (Preprocess): Preprocesamiento escogido por el usuario.
                                        Default: BOWPreprocess.
            params (dict): Diccionario con los hiperparámetros personalizados
                            que se a la instancia de MLLogisticRegression. La
                            lista de posibles hiperparámetros se encuentra a
                            continuación. Default: {}.

        params:
            penalty (str): Norma de penalización. Default: 'l2'.
            dual (bool): Formulación primal dual. Default: False.
            tol (float): Criterio de término. Default: 1e-4.
            C (float): Inverso de la ponderación de regularización.
                        Default 1.
            fit_intercept (bool): Especifica si una constante se debe agregar
                                    a la función de decisión. Default: True.
            intercept_scaling (float): Útil solo cuando el solver es
                                        "liblinear" y fit_intercept es True.
                                        En tal caso, x se transforma en
                                        [x, self.intercept_scaling].
            class_weight (dict): Pesos asociados a cada clase de la forma
                                    {class_label: weight}. Default: Todas las
                                    clases con mismo peso.
            random_state (int): Seed para replicar resultados. Default: None.
            solver (str): Algoritmo utilizado para resolver el problema
                            de optimización. Default: 'lbfgs'.
            max_iter (int): Número máximo de iteraciones tomadas por el
                            solver para converger. Default: 100.
            verbose (int): Nivel de especifidad que se quiere de la
                            información entregada. Default: 0. Cualquier
                            otro entero positivo agrega la información.
            warm_start (bool): True si se quiere partir desde el resultado
                                anterior en el fit. En otro caso, se comienza
                                desde cero. Default: False.
            l1_ratio: Elastic-Net mixing parameter. Default: None.
        """
        preprocess = kwargs.get('preprocess', None)
        if preprocess is None:
            preprocess = BOWPreprocess({'params':{}, 'tokenizer': []})
        super().__init__(preprocess)

        self.params = kwargs.get('params', {})
        self.log_r = OneVsRestClassifier(
            LogisticRegression(**self.params), n_jobs=-1)
        #self.schema = get_obj_schema(LogisticRegression())

    def fit(self, X, y):
        """
        Método que se usa para entrenar el modelo, no debe retornar nada.

        Args:
            X (array-like): Arreglo donde cada componente tiene un texto ya
                            preprocesado por preprocess.
            y (array-like): Arreglo donde cada componente tiene un arreglo
                            binario indicando si se presenta o no la label.
        """
        self.log_r.fit(X, y)

    def predict(self, X):
        """
        Método que se usa para predecir.

        Args:
            X (array-like): Arreglo donde cada componente tiene un texto ya
                            preprocesado por preprocess.

        Retorna matriz de adjacencia, que indica la pertenencia
        de las labels a las sentencias. En las filas se
        representan las sentencias (en orden), y en las
        columnas las etiquetas (en orden).

        """
        return self.log_r.predict(X)

    def predict_proba(self, X):
        """
        Entrega la probabilidad de que cada etiqueta pertenezca a las
        distintas sentencias.

        Args:
            X (array-like): Arreglo donde cada componente tiene un texto ya
                            preprocesado por preprocess.

        Retorna dataFrame, donde cada sentencia es representada
        en una fila, y en cada columna se representa una
        etiqueta. En la entrada (m,n) se puede observar la
        probabilidad de que en la m-ésima sentencia esté
        la n-ésima etiqueta.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return pd.DataFrame(self.log_r.predict_proba(X))

    def save(self, filename=None):
        """
        Método encargado de guardar la instancia entrenada del modelo.

        Args:
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

        Args:
            filename (str, optional): Nombre/path de destino desde donde
                                        se obtendrá el modelo entrenado.
                                        Default: None.

        Retorna modelo previamente guardado.
        """
        model = joblib.load(filename)
        return model

MODEL = MLLogisticRegression

if __name__ == "__main__":
    pass

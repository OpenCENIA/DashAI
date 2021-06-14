import logging
from abc import ABCMeta, abstractmethod

logger = logging.getLogger()

#version == 1.0.0
class Model(metaclass=ABCMeta):
    """
    Clase abstracta padre de los modelos.

    metaclass (ABCMeta): Metaclase. Default: ABCMeta.
    """
    def __init__(self, preprocess, *args, **kwargs):
        """
        Inicializa un modelo que hereda de esta clase.

        Args:
            preprocess (PreProcess): Preprocesamiento que se aplica.
        """
        self.prep = preprocess

    def preprocess(self, X):
        """
        Método que se usa para preprocesar los datos y, de esta forma,
        generar el input que recibe el modelo en fit, predict y predict_proba.

        X (array-like): Arreglo donde cada componente tiene un
                        texto plano que necesita ser preprocesado.
        """
        return self.prep.apply(X)

    @abstractmethod
    def fit(self, X, y):
        """
        Método que se usa para entrenar el modelo, no debe retornar nada.

        X (array-like): Arreglo donde cada componente tiene un texto ya
                        preprocesado por el preprocess.
        y (array-like): Arreglo donde cada componente tiene un arreglo de valores
                        binarios indicando si se prensentan, o no, las labels.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Método que se usa para predecir el X ingresado.

        X (array-like): Arreglo donde cada componente tiene un texto ya
                        preprocesado por el preprocess.
        """
        pass

    @abstractmethod
    def predict_proba(self, x):
        """
        X (array-like): Arreglo donde cada componente tiene un texto ya
                        preprocesado por el preprocess.

        Debe poder retornar un pd.DataFrame tipo:
        index        |     label        | probability
        id sentencia | numero del label | 0 <= p < 1
        """
        pass

    @abstractmethod
    def save(self, filename=None):
        """
        Guarda el modelo en archivos.

        filename (Str): Path donde quedará guardado el archivo.

        """
        pass


    @staticmethod
    def load(filename):
        """
        Carga el modelo desde los archivos.

        filename (Str): Path del archivo que contiene el modelo.
        """
        pass


if __name__ == "__main__":
    pass

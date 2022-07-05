import logging
from abc import ABCMeta, abstractmethod



# version == 1.0.0
class Model(metaclass=ABCMeta):
    """
    Abstract class of all machine learning models.
    The models must implement the save and load methods.
    """

    # TODO implement a check_params method to check the params
    #  using the JSON schema file.
    # TODO implement a method to check the initialization of TASK
    #  an task params variables.

    @abstractmethod
    def save(self, filename=None):
        """
        Stores an instance of a model.

        filename (Str): Indicates where to store the model,
        if filename is None, this method returns a bytes array with the model.
        """
        pass

    @staticmethod
    def load(filename):
        """
        Restores an instance of a model

        filename (Str): Indicates where the model was stored.
        """
        pass

    # def __init__(self, preprocess, *args, **kwargs):
    #     """
    #     Inicializa un modelo que hereda de esta clase.

    #     Args:
    #         preprocess (PreProcess): Preprocesamiento que se aplica.
    #     """
    #     self.prep = preprocess

    # def preprocess(self, X):
    #     """
    #     Método que se usa para preprocesar los datos y, de esta forma,
    #     generar el input que recibe el modelo en fit, predict y predict_proba

    #     X (array-like): Arreglo donde cada componente tiene un
    #                     texto plano que necesita ser preprocesado.
    #     """
    #     return self.prep.apply(X)

    # @abstractmethod
    # def fit(self, X, y):
    #     """
    #     Método que se usa para entrenar el modelo, no debe retornar nada.

    #     X (array-like): Arreglo donde cada componente tiene un texto ya
    #                     preprocesado por el preprocess.
    #     y (array-like): Arreglo donde cada componente tiene un arreglo de
    #                     valores binarios indicando si se prensentan,
    #                     o no, las labels.
    #     """
    #     pass

    # @abstractmethod
    # def predict(self, X):
    #     """
    #     Método que se usa para predecir el X ingresado.

    #     X (array-like): Arreglo donde cada componente tiene un texto ya
    #                     preprocesado por el preprocess.
    #     """
    #     pass

    # @abstractmethod
    # def predict_proba(self, x):
    #     """
    #     X (array-like): Arreglo donde cada componente tiene un texto ya
    #                     preprocesado por el preprocess.

    #     Debe poder retornar un pd.DataFrame tipo:
    #     index        |     label        | probability
    #     id sentencia | numero del label | 0 <= p < 1
    #     """
    #     pass

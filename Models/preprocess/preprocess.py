import logging
from abc import ABCMeta, abstractmethod

from Models.preprocess.tokenizer.tokenizer_cont import TokenizerContainer

logger = logging.getLogger()

#version == 1.0.0
class PreProcess(metaclass=ABCMeta):
    """
    Clase (abstracta) padre de los preprocesamientos.

    Args:
        metaclass (ABCMeta): Metaclase. Default: ABCMeta.
    """
    def __init__(self, dic):
        """
        dic (dict): Diccionario que contiene informacion para crear el
                    tokenizer container.
        """
        self.tokenizer_cont = TokenizerContainer(dic)

    @abstractmethod
    def apply(self, text):
        """
        Método abstracto que simboliza la ejecución del preprocesamiento.

        text (array-like): Array que contiene el input en formato de texto.
        """
        pass

if __name__ == "__main__":
    pass

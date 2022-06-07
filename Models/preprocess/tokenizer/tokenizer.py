import logging
from abc import ABCMeta, abstractmethod

logger = logging.getLogger()


# version == 1.0.0
class Tokenizer(metaclass=ABCMeta):
    """
    Clase (abstracta) padre de los tokenizadores.

    Args:
        metaclass (ABCMeta): Metaclase. Default: ABCMeta.
    """

    def __init__(self, **kwargs):
        """
        Inicializa un modelo que hereda de Tokenizer.
        """
        pass

    @abstractmethod
    def apply(self, text):
        """
        Retorna el texto tokenizado.

        Args:
            text (array-like): Array con el input que se quiere tokenizar,
                                en formato de texto.

        Returns:
            array-like: Array con el texto tokenizado.
        """
        pass


if __name__ == "__main__":
    pass

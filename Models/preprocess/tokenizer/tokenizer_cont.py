import logging

from Models.preprocess.tokenizer.null_tokenizer import NullTokenizer

logger = logging.getLogger()


# version: 1.0.0
class TokenizerContainer:
    """
    Contenedor de múltiples tokenizadores.
    """

    def __init__(self, lista):
        """
        Inicializa el contenedor.

        Args:
            lista (list): Lista de tokenizadores instanciados.
        """
        if not lista:
            self.container = [NullTokenizer()]
        else:
            self.container = lista

    def apply(self, text):
        """
        Retorna el texto tokenizado.

        Args:
            text (array-like): Array que contiene el input en formato de texto.

        Returns:
            array-like: Array que contiene el input tokenizado.
        """
        for tok in self.container:
            name = tok.__class__.__name__
            logger.debug(f"Comenzando tokenización {name}")
            text = tok.apply(text)
            logger.debug(f"Terminando tokenización {name}")
        return text


if __name__ == "__main__":
    pass

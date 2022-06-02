import re

from Models.preprocess.tokenizer.tokenizer import Tokenizer


# version: 1.0.0
class RemovePunctuations(Tokenizer):
    """
    Tokenizador para remover puntuaciones.

    Tokenizer: Clase padre de todos los tokenizadores implementados.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def apply(self, text):
        """
        Se aplica el tokenizador sobre el input.

        Args:
            text (array-like): Arreglo con los documentos de input en
                                formato texto.

        Returns:
            array-like: Arreglo de texto sin puntuaciones.
        """
        lista = [
            sent.translate(
                str.maketrans("", "", """!()-[]{};:'",<>./?@#$%^&*_~""")
            )
            for sent in text
        ]
        lista = [re.sub("“", "", sent) for sent in lista]
        lista = [re.sub("”", "", sent) for sent in lista]
        lista = [re.sub("…", "", sent) for sent in lista]
        lista = [" ".join(sent.split()) for sent in lista]
        return lista

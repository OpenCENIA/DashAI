import spacy

from Models.preprocess.tokenizer.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")


# version: 1.0.0
class Lemmatization(Tokenizer):
    """
    Lemmatization

    Tokenizer: Clase padre de todos los tokenizer implementados.
    """

    def __init__(self, **kwargs):
        """
        Proceso lingüístico en el que se llega a la raíz de la palabra.
        Utilizando técnica de lemmatización.
        """
        super().__init__()

    def apply(self, text):
        """
        Retorna el texto lematizado.

        Args:
            text (array-like): Array con el input que se quiere lematizar,
                                en formato de texto.

        Returns:
            array-like: Array con el texto lematizado.
        """
        return [" ".join([x.lemma_ for x in nlp(sent)]) for sent in text]


if __name__ == "__main__":
    pass

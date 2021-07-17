#import functions
import nltk
import spacy
from Models.preprocess.tokenizer.tokenizer import Tokenizer

def aux_stopwords(row, stopwords):
    """Remueve las stopwords de una fila.

    Args:
        row (pandas row): Fila a la cual se le remueven las stopwords.
        stopwords (list): Lista con las stopwords.

    Returns:
        pandas row: Fila modificada.
    """
    row = ' ' + row + ' '
    for word in stopwords:
        mod1 = ' ' + word + ' '
        mod2 = ' ' + word.capitalize() + ' '
        row = row.replace(mod1, ' ')
        row = row.replace(mod2, ' ')
    row = row.strip()
    return row

english_stopwords = nltk.corpus.stopwords.words("english")
nlp = spacy.load("en_core_web_sm")

#version: 1.0.0
class RemoveStopwords(Tokenizer):
    """
    Tokenizador para remover stopwords.

    Tokenizer: Clase padre de todos los tokenizadores implementados.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.stopwords = kwargs.get('stopwords', english_stopwords)

    def apply(self, text):
        """
        Se aplica el tokenizador sobre el input.

        Args:
            text (array-like): Arreglo con los documentos de input en
                                formato texto.

        Returns:
            list: Arreglo de texto sin stopwords.
        """
        return [aux_stopwords(x, self.stopwords) for x in text]

if __name__ == "__main__":
    pass
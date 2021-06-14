import functions
import nltk
import spacy
from preprocess.tokenizer.tokenizer import Tokenizer


spanish_stopwords = nltk.corpus.stopwords.words("spanish")
nlp = spacy.load('es_core_news_sm', disable=['ner', 'parser', 'tagger'])

#version: 1.0.0
class RemoveStopwords(Tokenizer):
    """
    Tokenizador para remover stopwords.

    Tokenizer: Clase padre de todos los tokenizadores implementados.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.stopwords = kwargs.get('stopwords', spanish_stopwords)

    def apply(self, text):
        """
        Se aplica el tokenizador sobre el input.

        Args:
            text (array-like): Arreglo con los documentos de input en
                                formato texto.

        Returns:
            list: Arreglo de texto sin stopwords.
        """
        return [functions.aux_stopwords(x, self.stopwords) for x in text]

if __name__ == "__main__":
    pass
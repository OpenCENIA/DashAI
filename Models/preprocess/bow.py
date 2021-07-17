import json
import logging

import spacy
from sklearn.feature_extraction.text import CountVectorizer

from Models.preprocess.preprocess import PreProcess

nlp = spacy.load("en_core_web_sm")
logger = logging.getLogger()

def identity_split(text):
    return text.split(' ')

#version == 1.0.0
class BOWPreprocess(PreProcess):
    """
    Vectorizador de tipo Bag of Words.

    Args:
        PreProcess: Clase padre de todos los preprocesamientos implementados.
    """
    with open('Models/parameters/preprocess_schemas/bow.json') as f:
        schema = json.load(f)

    def __init__(self, dic):
        """
        Inicializa una instancia de BOWPreprocess.

        Args:
            dic (dict): Diccionario con la información del preprocesamiento.
                        Las dos entradas del diccionario son:
                        
                        tokenizer (list): Lista de tokenizadores
                                            instanciados.
                        
                        params (dict): Contiene los hiperparámetros del
                                        preprocesamiento. Una lista detallada
                                        se encuentra a continuación.

        Hiperparámetros:
            encoding (str): Decodificación del texto. Default: utf-8.
            decode_error (str): Indica qué hacer si hay un error en la
                                decodificación. Los posibles valores son
                                {'strict', 'ignore', 'replace'}.
                                Default: 'strict'.
            strip_accents (str): Remueve acentos durante la normalización. Los
                                    posibles valores no nulos son 'ascii' y
                                'unicode'. Default: None.
            lowercase (bool): Convierte todos los caracteres a minúsculas.
                                Default: True.
            ngram_range (tuple): Cota inferior y superior para los rangos de
                                    n-gramas utilizados. Ejemplo: (1, 3) considera
                                    unigramas, bigramas y trigramas.
                                    Default: (1, 1).
            analyzer (str): Puede ser alguno de {'word', 'char', 'char_wb'}.
                            Indica si los n-gramas debiesen ser de palabras o
                            caracteres. Default: 'word'.
            max_df (float): En rango [0, 1]. Cuando se hacen n_gramas ignora
                            los términos que aparecen en los documentos con
                            una frecuencia superior a max_df. Default: 1.
            min_df (float): Análogo a max_df. Ignora términos con frecuencia en
                            documentos menor a min_df. Default: 1.
            max_features (int): Número máximo de elementos del vocabulario
                                ordenados por frecuencia de términos.
                                Default: None.
            binary (bool): Si es True, todos los conteos no nulos son seteados
                            iguales a 1. Default: False.
        """
        prep_kwargs = dic.get('params', {})
        tokenizer_kwargs = dic.get('tokenizers', None)
        super().__init__(tokenizer_kwargs)
        prep_kwargs['analyzer'] = prep_kwargs.get(
            'analyzer', 'word')
        prep_kwargs['tokenizer'] = prep_kwargs.get(
            'tokenizer', identity_split)
        prep_kwargs['ngram_range'] = prep_kwargs.get(
            'ngram_range', (1, 1))
        self.vectorizer = CountVectorizer(**prep_kwargs)
        self.fit_bool = True

    def apply(self, text):
        """
        Se aplica el preprocesamiento sobre el input.

        Args:
            text (array-like): Arreglo con los documentos de input en
                                formato texto.

        Returns:
            array-like: Arreglo con el input preprocesado.
        """
        text = self.tokenizer_cont.apply(text)
        if self.fit_bool:
            self.vectorizer.fit(text)
            self.fit_bool = False
        return self.vectorizer.transform(text)

if __name__ == "__main__":
    pass
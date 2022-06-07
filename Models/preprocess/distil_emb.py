import json
import logging

from sentence_transformers import SentenceTransformer

from Models.preprocess.preprocess import PreProcess

logger = logging.getLogger()


# version == 1.0.0
class DistilBertEmbedding(PreProcess):
    """
    Vectorizador de tipo Embedding.

    Args:
        PreProcess: Clase padre de todos los preprocesamientos implementados.
    """

    with open("Models/parameters/preprocess_schemas/distil.json") as f:
        schema = json.load(f)

    def __init__(self, dic):
        """
        Inicializa una instancia de Distil Preprocess.

        Args:
            dic (dict): Diccionario con la información del preprocesamiento.
                        Las dos entradas del diccionario son:

                        tokenizer (list): Lista de tokenizadores
                                            instanciados.

                        params (dict): Contiene los hiperparámetros del
                                        preprocesamiento. Una lista detallada
                                        se encuentra a continuación.
        """
        # prep_kwargs = dic.get("params", {})
        tokenizer_kwargs = dic.get("tokenizers", None)

        super().__init__(tokenizer_kwargs)
        self.model = SentenceTransformer(
            "paraphrase-distilroberta-base-v1", device="cpu"
        )

    def apply(self, text):
        """
        Se aplica el preprocesamiento sobre el input.

        Args:
            text (array-like): Arreglo con los documentos de input en
                                formato texto.

        Returns:
            array-like: Arreglo vectorial con el input preprocesado.
        """
        text = self.tokenizer_cont.apply(text)
        return self.model.encode(text)


if __name__ == "__main__":
    pass

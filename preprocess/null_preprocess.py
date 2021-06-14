import logging

from preprocess.preprocess import PreProcess

logger = logging.getLogger()

#version: 1.0.0
class NullPreprocess(PreProcess):
    """
    Args:
        PreProcess: Clase que no realiza cambios ni preprocesamiento. 
        Se utiliza para marcar la pauta de la estructura que deben tener los preprocesadores. 
    """
    def __init__(self, dic):
        prep_kwargs = dic.get('params', {})
        tokenizer_kwargs = dic.get('tokenizers', None)
        super().__init__(tokenizer_kwargs)

    def apply(self, text):
        logger.debug("Se ha aplicado NullPreprocess")
        text = self.tokenizer_cont.apply(text)
        return text

if __name__ == "__main__":
    pass

from preprocess.tokenizer.tokenizer import Tokenizer

class NullTokenizer(Tokenizer):
    """
    Tokenizador Nulo.

    Tokenizer: Clase padre de todos los tokenizer implementados.
    """
    def __init__(self, **kwargs):
        """
        Esta clase simula el no hacer ninguna tonkenizacion.
        Se utiliza para tener la arquitextura.
        """
        super().__init__()
        pass

    def apply(self, text):
        """
        Retorna el mismo texto de entrada.
        """
        return text

if __name__ == "__main__":
    pass
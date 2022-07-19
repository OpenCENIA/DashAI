from abc import ABCMeta, abstractmethod


# version == 1.0.0
class Tokenizer(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, text):
        """
        Convert text of type string in a token.
        """
        pass

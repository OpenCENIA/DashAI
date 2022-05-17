import logging
from abc import ABCMeta, abstractmethod
from Models.classes.model import Model
logger = logging.getLogger()

#version == 1.0.0
class Tokenizer(Model,metaclass=ABCMeta):

    def tok(self, s):
        return s
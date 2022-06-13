import logging
from abc import ABCMeta
logger = logging.getLogger()

#version == 1.0.0
class Tokenizer(metaclass=ABCMeta):

    def tok(self, s):
        return s
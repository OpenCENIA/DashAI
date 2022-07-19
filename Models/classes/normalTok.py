import json
from nltk.tokenize import word_tokenize
from Models.classes.tokenizer import Tokenizer


class NormalTokenizer(Tokenizer):
    MODEL = "normalTok"
    with open(f"Models/parameters/models_schemas/{MODEL}.json") as f:
        SCHEMA = json.load(f)
    def __call__(self, text):
        return word_tokenize(text)
import json

from Models.classes.tokenizer import Tokenizer


class NormalTokenizer(Tokenizer):
    MODEL = "normalTok"
    with open(f"Models/parameters/models_schemas/{MODEL}.json") as f:
        SCHEMA = json.load(f)

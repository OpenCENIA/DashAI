import json
from nltk.tokenize import TweetTokenizer
from Models.classes.tokenizer import Tokenizer


class TweetTokenizer(Tokenizer):
    MODEL = "tweetTok"
    with open(f"Models/parameters/models_schemas/{MODEL}.json") as f:
        SCHEMA = json.load(f)
    def __init__(self):
        self.tokenizer = TweetTokenizer()
    
    def __call__(self, text):
        return self.tokenizer.tokenize(text)

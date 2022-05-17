from Models.classes.tokenizer import Tokenizer
import json

class TweetTokenizer(Tokenizer):
    TASK = ["TextClassificationSimpleTask"]
    MODEL = "tweetTok"
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)
from Models.classes.sklearnLikeModel import SkleanLikeModel
from Models.classes.textClassificationModel import TextClassificationModel
import json

class NumericalWrapperForText(SkleanLikeModel ,TextClassificationModel):
    """
    Wrapper for TextClassificationTask, that uses a numericClassificationModel and
    a tokenizer to classify text.
    """
    MODEL = "numericalwrapperfortext"
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)
    
    def __init__(self, numeric_classifier, tokenizer) -> None:
        self.classifier = numeric_classifier
        self.tokenizer = tokenizer

    def fit(self, x, y):
        self.classifier.fit(self.tokenizer.tokenize(x), y)
    
    def predict(self, x):
        return self.classifier.predict(self.tokenizer.tokenize(x))

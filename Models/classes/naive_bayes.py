from sklearn.naive_bayes import GaussianNB
from Models.classes.model import Model

import json
from io import BytesIO
import joblib

class NaiveBayes(Model, GaussianNB):
    """
    """
    MODEL = "naivebayes"
    TASK = ["TextClassificationSimpleTask"]
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)
    
    def save(self, filename=None):

        if filename is None:
            bytes_container = BytesIO()
            joblib.dump(self, bytes_container)
            bytes_container.seek(0)
            return bytes_container.read()
        else:
            joblib.dump(self, filename)

    @staticmethod
    def load(filename):

        model = joblib.load(filename)
        return model
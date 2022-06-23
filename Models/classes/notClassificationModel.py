from sklearn.svm import SVC
from Models.classes.model import Model

import json
from io import BytesIO
import joblib

class notClassificationModel(Model):
    """
    Just a dummy model to check that text classification tasks do not 
    consider it on their compatible models
    """
    MODEL = "notClassificationModel"
    TASK = ["AnotherTask"]
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)
    
    def save(self, filename=None):
        pass

    @staticmethod
    def load(filename):
        pass
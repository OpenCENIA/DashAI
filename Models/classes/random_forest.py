from sklearn.ensemble import RandomForestClassifier
from Models.classes.model import SkleanLikeModel
import json


class RandomForest(SkleanLikeModel, RandomForestClassifier):
    """
    """
    MODEL = "randomforest"
    TASK = ["TextClassificationSimpleTask"]
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)

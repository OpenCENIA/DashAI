from sklearn.ensemble import RandomForestClassifier
from Models.classes.sklearn_like_model import SkleanLikeModel
from Models.classes.numericClassificationModel import NumericClassificationModel
import json


class RandomForest(SkleanLikeModel, NumericClassificationModel, RandomForestClassifier):
    """
    """
    MODEL = "randomforest"
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)

from sklearn.naive_bayes import GaussianNB
from Models.classes.sklearnLikeModel import SkleanLikeModel
from Models.classes.numericClassificationModel import NumericClassificationModel
import json


class NaiveBayes(SkleanLikeModel, NumericClassificationModel, GaussianNB):
    """
    """
    MODEL = "naivebayes"
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)

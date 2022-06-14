from sklearn.naive_bayes import GaussianNB
from Models.classes.model import SkleanLikeModel
import json


class NaiveBayes(SkleanLikeModel, GaussianNB):
    """
    """
    MODEL = "naivebayes"
    TASK = ["TextClassificationSimpleTask"]
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)

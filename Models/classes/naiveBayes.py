import json

from sklearn.naive_bayes import GaussianNB

from Models.classes.numericClassificationModel import NumericClassificationModel
from Models.classes.sklearnLikeModel import SkleanLikeModel


class NaiveBayes(SkleanLikeModel, NumericClassificationModel, GaussianNB):
    """ """

    MODEL = "naivebayes"
    with open(f"Models/parameters/models_schemas/{MODEL}.json") as f:
        SCHEMA = json.load(f)

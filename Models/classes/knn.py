from sklearn.neighbors import KNeighborsClassifier
from Models.classes.sklearn_like_model import SkleanLikeModel
from Models.classes.numericClassificationModel import NumericClassificationModel
import json


class KNN(SkleanLikeModel, NumericClassificationModel, KNeighborsClassifier):
    """
    K Nearest Neighbors is a supervized classification method, 
    that determines the probability that an element belongs to 
    a certain class, considering its k nearest neighbors. 
    """
    MODEL = "knn"
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)

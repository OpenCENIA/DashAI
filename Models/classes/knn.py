from sklearn.neighbors import KNeighborsClassifier
from Models.classes.model import SkleanLikeModel
import json


class KNN(SkleanLikeModel, KNeighborsClassifier):
    """
    K Nearest Neighbors is a supervized classification method, 
    that determines the probability that an element belongs to 
    a certain class, considering its k nearest neighbors. 
    """
    MODEL = "knn"
    TASK = ["TextClassificationSimpleTask"]
    with open(f'Models/parameters/models_schemas/{MODEL}.json') as f:
        SCHEMA = json.load(f)

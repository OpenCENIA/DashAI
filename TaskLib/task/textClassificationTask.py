from TaskLib.task.numericClassificationTask import NumericClassificationTask
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV

from TaskLib.task.taskMain import Task


class TextClassificationTask(Task):
    """
    Abstarct class for text classification tasks.
    Never use this class directly.
    """
    NAME : str = "TextClassificationTask"

    # def get_parameters_structure(self) -> dict:
    #     """
    #     The available params are LABEL and INSTANCE, the expected
    #     values for both are SINGLE and MULTI, and the readable form
    #     for both are Single and Multi
    #     """
    #     structure = {}
    #     structure["LABEL"] = [
    #         ["SINGLE",'Single'],
    #         ["MULTI",'Multi']
    #     ]
    #     structure["INSTANCE"] = [
    #         ["SINGLE",'Single'],
    #         ["MULTI",'Multi']
    #     ]
    #     return structure

    # def config(self, params : dict) -> None:
    #     """
    #     The params dictionary must have LABEL and INSTANCE keys
    #     """
    #     self.label : str = params.get("LABEL")
    #     self.instance : str = params.get("INSTANCE")

    # def get_parameters(self) -> dict:
    #     params = {}

    #     params["LABEL"] = self.label
    #     params["INSTANCE"] = self.instance

    #     return params

    # def get_compatible_models(self) -> list:

    #     compatible_models : list = []
    #     for modelName in get_available_models():
    #         modelClass = globals().get(modelName)
    #         if "TEXT" in modelClass.TASK:
    #             if modelClass.INSTANCE == self.instance and
    #             modelClass.LABEL == self.label:
    #                 compatible_models.append(modelName)

    #     return compatible_models

    def set_executions(self, models: list, params: list):

        pass
        # self.gridExecutions : list = []
        # for i in range(len(models)):
        #     actualExecution = globals().get(models[i])()
        #     grid = GridSearchCV(actualExecution, params[i], cv=2)
        #     self.gridExecutions.append(grid)
    
    def run_experiments(self, input_data: dict):

        train_x, train_y, test_x, test_y = parse_input(input_data)

        count_vect = CountVectorizer()
        count_vect.fit(train_x)
        count_vect.fit(test_x)

        X_train_counts = count_vect.transform(train_x)
        X_test_counts = count_vect.transform(test_x)

        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_test_counts)

        prep_train_x = tf_transformer.transform(X_train_counts)
        prep_test_x = tf_transformer.transform(X_test_counts)

        # self.experimentResults = {}

        # for grid in self.gridExecutions:
        #     grid.fit(prep_train_x, train_y)

        #     trainResults = grid.score(prep_train_x, train_y)
        #     testResults = grid.score(prep_test_x, test_y)
        #     parameters = grid.best_params_      
        #     execution = grid.best_estimator_
        #     executionBytes = execution.save()

        #     self.experimentResults[execution.MODEL] = {
        #         "train_results" : trainResults,
        #         "test_results" : testResults,
        #         "parameters" : parameters,
        #         "executionBytes" : executionBytes
        #     }



def parse_input(input_data):
    # TODO reshape only if input is 1D
    x_train = np.array(input_data["train"]["x"])
    y_train = np.array(input_data["train"]["y"])
    x_test = np.array(input_data["test"]["x"])
    y_test = np.array(input_data["test"]["y"])

    return x_train, y_train, x_test, y_test

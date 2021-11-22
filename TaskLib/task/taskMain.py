from abc import ABC,abstractmethod
from sklearn.model_selection import GridSearchCV
from Models.classes import *
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# TODO implement classes for tokenizer

class Task(ABC):
    
    @abstractmethod
    def config(self, params : dict) -> None: 
        """
        This method recieves a dictionary with important parameters of the task.
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """
        This method returns a dictionary with all important parameters of the task.
        """
        pass

    @abstractmethod
    def get_compatible_models(self) -> list:
        """
        This method provides all task compatible models that are currently available.

        Return a list of string with the names of the models.
        """
        pass
    
    def set_executions(self, models : list, params : list) -> None:
        """
        This method configures one execution per model in models with the parameters
        in the params[model] dictionary.

        The executions were temporaly save in self.gridExecutions.
        """
        self.gridExecutions : list = []
        for i in range(len(models)):
            actualExecution = globals().get(models[i])()
            grid = GridSearchCV(actualExecution, params[i], cv=2)
            self.gridExecutions.append(grid)
    
    def run_experiments(self, input_data : dict):
        """
        This method train all the executions in self.executions with the data in input_data.

        The input_data dictionary must have train, validation and test keys to perform the training.

        The test results were temporaly save in self.experimentResults.
        """
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

        self.experimentResults = {}

        for grid in self.gridExecutions:
            grid.fit(prep_train_x, train_y)

            trainResults = grid.score(prep_train_x, train_y)
            testResults = grid.score(prep_test_x, test_y)
            parameters = grid.best_params_      
            execution = grid.best_estimator_
            executionBytes = execution.save()

            self.experimentResults[execution.MODEL] = {
                "train_results" : trainResults,
                "test_results" : testResults,
                "parameters" : parameters,
                "executionBytes" : executionBytes
            }
    

def parse_input(input_data):
    #TODO reshape only if input is 1D
    x_train = np.array(input_data["train"]["x"])
    y_train = np.array(input_data["train"]["y"])
    x_test = np.array(input_data["test"]["x"])
    y_test = np.array(input_data["test"]["y"])

    return x_train, y_train, x_test, y_test
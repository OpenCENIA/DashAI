from abc import ABC,abstractmethod
from Models.classes.getters import filter_by_parent
import numpy as np

class Task(ABC):
    """
    Task is an abstract class for all the Task implemented in the framework.
    Never use this class directly.
    """

    # task name, present in the compatible models
    NAME : str = ""
    compatible_models :list = []
    
    def __init__(self):
        self.set_compatible_models()

    def set_compatible_models(self) -> None:

        task_name = self.NAME if self.NAME else Exception("Need specify task name")
        model_class_name = f"{task_name[:-4]}Model"
        self.compatible_models = filter_by_parent(model_class_name)

    def __init__(self):

        self.set_compatible_models()

    def set_compatible_models(self) -> None:

        task_name = self.NAME if self.NAME else Exception("Need specify task name")
        model_class_name = f"{task_name[:-4]}Model"
        self.compatible_models = filter_by_parent(model_class_name)

    def get_compatible_models(self) -> list:

        return self.compatible_models

    def set_executions(self, models : list, params : dict) -> None:
        """
        This method configures one execution per model in models with the parameters
        in the params[model] dictionary.
        The executions were temporaly save in self.executions.
        """
        # TODO
        # Generate a Grid to search the best model
        self.executions : list = []
        for model in models:
            actualExecution = self.compatible_models[model]
            self.executions.append(actualExecution(**params[model]))
    
    def run_experiments(self, input_data : dict):
        """
        This method train all the executions in self.executions with the data in input_data.
        The input_data dictionary must have train and test keys to perform the training.
        The test results were temporaly save in self.experimentResults.
        """        
        x_train = np.array(input_data["train"]["x"])
        y_train = np.array(input_data["train"]["y"])
        x_test = np.array(input_data["test"]["x"])
        y_test = np.array(input_data["test"]["y"])

        categories = []
        for cat in y_train:
            if cat not in categories:
                categories.append(cat)
        for cat in y_test:
            if cat not in categories:
                categories.append(cat)

        numeric_y_train = []
        for sample in y_train:
            numeric_y_train.append(categories.index(sample))
        numeric_y_test = []
        for sample in y_test:
            numeric_y_test.append(categories.index(sample))

        self.experimentResults = {}

        for execution in self.executions:
            execution.fit(x_train, numeric_y_train)

            trainResults = execution.score(x_train, numeric_y_train)
            testResults = execution.score(x_test, numeric_y_test)
            parameters = execution.get_params()
            executionBytes = execution.save()

            self.experimentResults[execution.MODEL] = {
                "train_results" : trainResults,
                "test_results" : testResults,
                "parameters" : parameters,
                "executionBytes" : executionBytes
            }
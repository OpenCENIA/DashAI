from abc import ABC,abstractmethod
from Models.classes.getters import filter_by_parent

# from sqlalchemy import inspect
# from Models.classes import *
# from inspect import isclass
# from pkgutil import iter_modules
# from pathlib import Path
# from importlib import import_module
# from Models import classes


class Task(ABC):
    """
    Task is an abstract class for all the Task implemented in the framework.
    Never use this class directly.
    """

    # task name, present in the compatible models
    NAME : str = ""
    compatible_models :list = []

    # @abstractmethod
    # def get_parameters_structure(self) -> dict:
    #     """
    #     This method provides a dictionary with all the task's configurations,
    #     for each parameter, there is a list of posible values and the readable form.
    #     The structure is
    #     {
    #         ...,
    #         "param_i": [
    #             ...,
    #             [name_j, read_form_j],
    #             ...
    #         ],
    #         ...
    #     }
    #     """
    
    # @abstractmethod
    # def config(self, params : dict) -> None: 
    #     """
    #     This method recieves a dictionary with important parameters of the task.
    #     """
    #     pass
    
    # @abstractmethod
    # def get_parameters(self) -> dict:
    #     """
    #     This method returns a dictionary with all important parameters of the task.
    #     """
    #     pass

    def __init__(self):

        self.set_compatible_models()

    def set_compatible_models(self) -> None:

        task_name = self.NAME if self.NAME else Exception("Need specify task name")
        model_class_name = f"{task_name[:-4]}Model"
        self.compatible_models = filter_by_parent(model_class_name)

    def get_compatible_models(self) -> list:

        return self.compatible_models

    
    
    @abstractmethod
    def set_executions(self, models : list, params : list) -> None:
        """
        This method configures one execution per model in models with the parameters
        in the params[model] dictionary.

        The executions were temporaly save in self.gridExecutions.
        """
        pass
    
    @abstractmethod
    def run_experiments(self, input_data : dict):
        """
        This method train all the executions in self.executions with the data in input_data.

        The input_data dictionary must have train, validation and test keys to perform the training.

        The test results were temporaly save in self.experimentResults.
        """
        pass
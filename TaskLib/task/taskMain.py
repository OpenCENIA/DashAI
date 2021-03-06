from abc import ABC, abstractmethod

from Models.classes import get_available_models


class Task(ABC):
    """
    Task is an abstract class for all the Task implemented in the framework.
    Never use this class directly.
    """

    # task name, present in the compatible models
    NAME: str = ""

    # @abstractmethod
    # def get_parameters_structure(self) -> dict:
    #     """
    #     This method provides a dictionary with all the task's configurations,
    #     for each parameter, there is a list of posible values and the
    #     readable form.
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
    #     This method recieves a dictionary with important parameters of
    #     the task.
    #     """
    #     pass

    # @abstractmethod
    # def get_parameters(self) -> dict:
    #     """
    #     This method returns a dictionary with all important parameters of
    #     the task.
    #     """
    #     pass

    def get_compatible_models(self) -> list:
        """
        This method provides all task compatible models that are currently
        available.
        Return a list of string with the names of the models.
        """
        compatible_models: list = []
        for modelName in get_available_models():
            modelClass = globals().get(modelName)
            if self.NAME in modelClass.TASK:
                compatible_models.append(modelName)

        return compatible_models

    @abstractmethod
    def set_executions(self, models: list, params: list) -> None:
        """
        This method configures one execution per model in models with
        the parameters in the params[model] dictionary.

        The executions were temporaly save in self.gridExecutions.
        """
        pass

    @abstractmethod
    def run_experiments(self, input_data: dict):
        """
        This method train all the executions in self.executions
        with the data in input_data.

        The input_data dictionary must have train, validation
        and test keys to perform the training.

        The test results were temporaly save in self.experimentResults.
        """
        pass

from abc import ABC,abstractmethod

class Task(ABC):
    
    @abstractmethod
    def config(self, params : dict) -> None: 
        """
        This method recieves a dictionary with important parameters of the task.
        """
        pass

    @abstractmethod
    def get_compatible_models(self) -> list:
        """
        This method provides all task compatible models that are currently available.

        Return a list of string with the names of the models.
        """
        pass
    
    @abstractmethod
    def set_executions(self, models : list, params : list):
        """
        This method configures one execution per model in models with the parameters
        in the params[model] dictionary.

        The executions were temporaly save in self.executions.
        """
        pass
    
    @abstractmethod
    def run_experiments(self, input_data : dict):
        """
        This method train all the executions in self.executions with the data in input_data.

        The input_data dictionary must have train, validation and test keys to perform the training.

        The test results were temporaly save in self.results.
        """
        pass
    
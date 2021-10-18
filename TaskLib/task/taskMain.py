from abc import ABC,abstractmethod
from sklearn.model_selection import GridSearchCV

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
    
    def set_executions(self, models : list, params : list) -> None:
        """
        This method configures one execution per model in models with the parameters
        in the params[model] dictionary.

        The executions were temporaly save in self.gridExecutions.
        """
        self.gridExecutions : list = []
        for i in range(len(models)):
            actualExecution = globals().get(models[i])()
            grid = GridSearchCV(actualExecution, params[i])
            self.gridExecutions.append(grid)
    
    def run_experiments(self, input_data : dict):
        """
        This method train all the executions in self.executions with the data in input_data.

        The input_data dictionary must have train, validation and test keys to perform the training.

        The test results were temporaly save in self.experimentResults.
        """
        train_x, train_y, test_x, test_y = parse_input(input_data)

        self.experimentResults = {}

        for grid in self.gridExecutions:
            grid.fit(train_x, train_y)

            trainResults = grid.best_score
            testResults = grid.score(test_x, test_y)
            parameters = grid.best_params      
            execution = grid.best_estimator
            executionBytes = execution.save()

            self.experimentResults[execution.MODEL] = {
                "train_results" : trainResults,
                "test_results" : testResults,
                "parameters" : parameters,
                "executionBytes" : executionBytes
            }
    

def parse_input(input_data):
    
    x_train = input_data["train"]["x"]
    y_train = input_data["train"]["y"]
    x_test = input_data["test"]["x"]
    y_test = input_data["test"]["y"]

    return x_train, y_train, x_test, y_test
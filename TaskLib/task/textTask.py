from TaskLib.task.taskMain import Task
from Models.classes import get_available_models

class TextTask(Task):

    def config(self, params: dict) -> None:
        """
        The params dictionary must have LABEL and INSTANCE keys
        """
        self.label : str = params.get("LABEL")
        self.instance : str = params.get("INSTANCE")
    
    def get_compatible_models(self) -> list:
        
        compatible_models : list = []
        for modelName in get_available_models():
            modelClass = globals().get(modelName)
            if modelClass.INSTANCE == self.instance and modelClass.LABEL == self.label:
                compatible_models.append(modelName)
        
        return compatible_models
    
    def set_executions(self, models: list, params: list):
        # TODO use greadSeach, not the model
        self.executions : list = []
        for modelName in models:
            actualExecution = globals().get(modelName)(params[modelName])
            self.executions.append(actualExecution)
    
    def run_experiments(self, input_data: dict):
        # TODO generate a function to parse de input_data
        for exec in self.executions:
            # fit the grid
            # get the results and the parameters
            pass
    
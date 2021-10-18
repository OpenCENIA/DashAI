from TaskLib.task.taskMain import Task
from Models.classes import get_available_models
from sklearn.model_selection import GridSearchCV

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

        super().set_executions(models, params)
    
    def run_experiments(self, input_data: dict):
        
        super().run_experiments(input_data)
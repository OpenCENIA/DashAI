from TaskLib.task.taskMain import Task
from Models.classes import *
class TextClassificationTask(Task):

    def get_parameters_structure(self) -> dict:
        """
        The available params are LABEL and INSTANCE, the expected 
        values for both are SINGLE and MULTI, and the readable form  
        for both are Single and Multi
        """
        structure = {}
        structure["LABEL"] = [
            ["SINGLE",'Single'],
            ["MULTI",'Multi']
        ]
        structure["INSTANCE"] = [
            ["SINGLE",'Single'],
            ["MULTI",'Multi']
        ]
        return structure

    def config(self, params : dict) -> None:
        """
        The params dictionary must have LABEL and INSTANCE keys
        """
        self.label : str = params.get("LABEL")
        self.instance : str = params.get("INSTANCE")
    
    def get_parameters(self) -> dict:
        params = {}

        params["LABEL"] = self.label
        params["INSTANCE"] = self.instance

        return params
    
    def get_compatible_models(self) -> list:
        
        compatible_models : list = []
        for modelName in get_available_models():
            modelClass = globals().get(modelName)
            if "TEXT" in modelClass.TASK:
                if modelClass.INSTANCE == self.instance and modelClass.LABEL == self.label:
                    compatible_models.append(modelName)
        
        return compatible_models
    
    def set_executions(self, models: list, params: list):

        super().set_executions(models, params)
    
    def run_experiments(self, input_data: dict):
        
        super().run_experiments(input_data)
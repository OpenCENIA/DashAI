from TaskLib.task.taskMain import Task

class NumericClassificationTask(Task):
    """
    Abstarct class for text classification tasks.
    Never use this class directly.
    """
    NAME : str = "NumericClassificationTask"

    def set_executions(self, models: list, params: list):
        pass

    def run_experiments(self, input_data: dict):
        pass
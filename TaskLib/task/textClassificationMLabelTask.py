from TaskLib.task.textClassificationTask import TextClassificationTask

class TextClassificationMLabelTask(TextClassificationTask):
    """
    Sofisticated task, that expect to receives a dataset with 
    more than one label per example.
    """

    NAME : str = "TextClassificationMLabelTask"
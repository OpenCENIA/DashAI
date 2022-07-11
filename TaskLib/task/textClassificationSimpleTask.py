from TaskLib.task.textClassificationTask import TextClassificationTask


class TextClassificationSimpleTask(TextClassificationTask):
    """
    Most common text classification task, the input of this task
    is a dataset with only one document and one label per example.
    """

    NAME: str = "TextClassificationSimpleTask"

    def __init__(self):
        super().__init__(self.NAME)

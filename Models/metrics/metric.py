from abc import ABCMeta, abstractmethod


class Metric(metaclass=ABCMeta):
    def __call__(self, y_true, y_pred):
        return self.apply(y_true, y_pred)

    @abstractmethod
    def apply(self, y_true, y_pred):
        pass

from sklearn.metrics import accuracy_score
from metrics.metric import Metric


class Accuracy(Metric):
    
    name = 'accuracy'

    def apply(self,y_true, y_pred):
        output = {}
        for i in range(y_true.shape[1]):
            acc = accuracy_score(y_true[:, i], y_pred[:, i])
            output[str(i)] = acc
        global_acc = accuracy_score(y_true, y_pred)
        output['global'] = global_acc

        return output

METRIC = Accuracy()
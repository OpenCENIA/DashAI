from sklearn.metrics import precision_score
from metrics.metric import Metric

class Precision(Metric):
    
    name = 'precision'

    def apply(self,y_true, y_pred):
        precision_labels = precision_score(y_true, y_pred, average=None, zero_division=1)
        macro = precision_score(y_true, y_pred, average='macro', zero_division=1)
        micro = precision_score(y_true, y_pred, average='micro', zero_division=1)
        weighted = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        samples = precision_score(y_true, y_pred, average='samples', zero_division=1)

        output = {}
        for i in range(len(precision_labels)):
            output[str(i)] = precision_labels[i]

        output['macro avg'] = macro
        output['micro avg'] = micro
        output['weighted avg'] = weighted
        output['samples avg'] = samples

        return output

METRIC = Precision()

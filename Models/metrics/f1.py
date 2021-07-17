from sklearn.metrics import f1_score
from Models.metrics.metric import Metric

class F1(Metric):
    
    METRIC = 'f1'

    def apply(self,y_true, y_pred):
        f1_labels = f1_score(y_true, y_pred, average=None, zero_division=1)
        macro = f1_score(y_true, y_pred, average='macro', zero_division=1)
        micro = f1_score(y_true, y_pred, average='micro', zero_division=1)
        weighted = f1_score(y_true, y_pred, average='weighted', zero_division=1)
        samples = f1_score(y_true, y_pred, average='samples', zero_division=1)

        output = {}
        for i in range(len(f1_labels)):
            output[str(i)] = f1_labels[i]

        output['macro avg'] = macro
        output['micro avg'] = micro
        output['weighted avg'] = weighted
        output['samples avg'] = samples

        return output

METRIC = F1()
from sklearn.metrics import recall_score
from Models.metrics.metric import Metric

class Recall(Metric):
    
    METRIC = 'recall'

    def apply(self,y_true, y_pred):
        recall_labels = recall_score(y_true, y_pred, average=None, zero_division=1)
        macro = recall_score(y_true, y_pred, average='macro', zero_division=1)
        micro = recall_score(y_true, y_pred, average='micro', zero_division=1)
        weighted = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        samples = recall_score(y_true, y_pred, average='samples', zero_division=1)

        output = {}
        for i in range(len(recall_labels)):
            output[str(i)] = recall_labels[i]

        output['macro avg'] = macro
        output['micro avg'] = micro
        output['weighted avg'] = weighted
        output['samples avg'] = samples

        return output

METRIC = Recall()
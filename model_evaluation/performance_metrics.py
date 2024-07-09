
# root/model_evaluation/performance_metrics.py
# Implements various model evaluation metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

class PerformanceMetrics:
    def __init__(self):
        pass

    def classification_metrics(self, y_true, y_pred, y_prob=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        return metrics

    def regression_metrics(self, y_true, y_pred):
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def custom_metric(self, y_true, y_pred):
        # Implement your custom metric here
        pass

if __name__ == '__main__':
    metrics = PerformanceMetrics()
    
    # Example usage for classification
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 1, 1]
    y_prob = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.3, 0.5, 0.2],
              [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.6, 0.3]]
    
    class_metrics = metrics.classification_metrics(y_true, y_pred, y_prob)
    print("Classification Metrics:", class_metrics)
    
    # Example usage for regression
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    
    reg_metrics = metrics.regression_metrics(y_true, y_pred)
    print("Regression Metrics:", reg_metrics)

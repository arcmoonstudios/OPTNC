# root/model_interpretation/feature_importance.py
# Implements methods for calculating and visualizing feature importance

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap

class FeatureImportance:
    def __init__(self, model=None):
        self.model = model

    def get_feature_importance(self, X, y, method='permutation', n_repeats=10):
        if method == 'permutation':
            perm_importance = permutation_importance(self.model, X, y, n_repeats=n_repeats)
            return perm_importance.importances_mean
        elif method == 'built-in':
            return self.model.feature_importances_
        else:
            raise ValueError("Unsupported method. Use 'permutation' or 'built-in'.")

    def plot_feature_importance(self, importances, feature_names):
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances')
        plt.barh(range(len(importances)), importances[indices])
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    def shap_analysis(self, X):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)

if __name__ == '__main__':
    # Example usage
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
    
    # For classification
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    fi_clf = FeatureImportance(clf)
    importances_clf = fi_clf.get_feature_importance(X, y, method='permutation')
    fi_clf.plot_feature_importance(importances_clf, feature_names)
    fi_clf.shap_analysis(X)
    
    # For regression
    y_reg = np.random.rand(100)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y_reg)
    
    fi_reg = FeatureImportance(reg)
    importances_reg = fi_reg.get_feature_importance(X, y_reg, method='built-in')
    fi_reg.plot_feature_importance(importances_reg, feature_names)

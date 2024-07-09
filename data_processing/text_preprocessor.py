# root/model_optimization/hyperparameter_tuning.py
# Implements hyperparameter tuning techniques

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class HyperparameterTuner:
    def __init__(self, model, param_grid, cv=5, scoring='accuracy'):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def grid_search(self, X, y):
        grid_search = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring=self.scoring)
        grid_search.fit(X, y)
        return grid_search.best_params_, grid_search.best_score_

    def random_search(self, X, y, n_iter=10):
        random_search = RandomizedSearchCV(self.model, self.param_grid, n_iter=n_iter, cv=self.cv, scoring=self.scoring)
        random_search.fit(X, y)
        return random_search.best_params_, random_search.best_score_

if __name__ == '__main__':
    # Example usage
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    tuner = HyperparameterTuner(model, param_grid)
    
    best_params_grid, best_score_grid = tuner.grid_search(X, y)
    print("Grid Search - Best params:", best_params_grid)
    print("Grid Search - Best score:", best_score_grid)
    
    best_params_random, best_score_random = tuner.random_search(X, y)
    print("Random Search - Best params:", best_params_random)
    print("Random Search - Best score:", best_score_random)

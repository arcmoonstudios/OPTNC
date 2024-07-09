
# root/data_processing/feature_engineering.py
# Implements various feature engineering techniques

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineer:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.selector = SelectKBest(score_func=f_classif)

    def create_polynomial_features(self, X):
        return self.poly.fit_transform(X)

    def scale_features(self, X):
        return self.scaler.fit_transform(X)

    def perform_pca(self, X, n_components=None):
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        self.pca.n_components = n_components
        return self.pca.fit_transform(X)

    def select_best_features(self, X, y, k='all'):
        return self.selector.fit_transform(X, y)

    def create_time_features(self, df, date_column):
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['quarter'] = df[date_column].dt.quarter
        return df

    def create_lag_features(self, series, lag_list):
        df = pd.DataFrame(series)
        for lag in lag_list:
            df[f'lag_{lag}'] = df.shift(lag)
        return df

    def bin_continuous_variable(self, series, n_bins=5, strategy='uniform'):
        return pd.cut(series, bins=n_bins, labels=False)

if __name__ == '__main__':
    engineer = FeatureEngineer()
    
    # Example usage
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    poly_features = engineer.create_polynomial_features(X)
    scaled_features = engineer.scale_features(X)
    pca_features = engineer.perform_pca(X, n_components=3)
    selected_features = engineer.select_best_features(X, y, k=3)
    
    print("Polynomial features shape:", poly_features.shape)
    print("Scaled features mean:", scaled_features.mean())
    print("PCA features shape:", pca_features.shape)
    print("Selected features shape:", selected_features.shape)

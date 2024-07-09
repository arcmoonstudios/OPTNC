# root/data_processing/preprocessor.py
# Handles data preprocessing steps before feeding into the neural network.

from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    def process(self, data):
        # Assuming data is a list of dictionaries or a 2D list
        # This is a simplified example and may need to be adapted based on your specific data structure
        numerical_features = []
        categorical_features = []

        for item in data:
            num_feat = []
            cat_feat = []
            for value in item.values() if isinstance(item, dict) else item:
                if isinstance(value, (int, float)):
                    num_feat.append(value)
                else:
                    cat_feat.append(str(value))
            numerical_features.append(num_feat)
            categorical_features.append(cat_feat)

        # Scale numerical features
        scaled_numerical = self.scaler.fit_transform(numerical_features)

        # Encode categorical features
        encoded_categorical = self.encoder.fit_transform(categorical_features)

        # Combine numerical and categorical features
        processed_data = np.hstack((scaled_numerical, encoded_categorical))

        return processed_data

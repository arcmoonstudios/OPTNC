# root/models/neural_model.py
# Defines and manages the neural network model.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class NeuralModel:
    def __init__(self, input_shape, output_shape):
        self.model = self._build_model(input_shape, output_shape)

    def _build_model(self, input_shape, output_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(output_shape, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load(cls, filepath):
        loaded_model = tf.keras.models.load_model(filepath)
        instance = cls(loaded_model.input_shape[1], loaded_model.output_shape[1])
        instance.model = loaded_model
        return instance

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib


class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass


class RandomForest(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        X_flatten = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_flatten, y_train)

    def predict(self, x_test):
        X_flatten = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(X_flatten)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class FeedForwardNeuralNetwork(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=64)

    def predict(self, x_test):
        predictions = self.model.predict(x_test, verbose=0)
        return np.argmax(predictions, axis=1)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


class ConvolutionalNeuralNetwork(MnistClassifierInterface):
    def __init__(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=64)

    def predict(self, X_test):
        predictions = self.model.predict(X_test, verbose=0)
        return np.argmax(predictions, axis=1)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


class MnistClassifier:
    def __init__(self, algorithm):
        models_map = {
            'cnn': ConvolutionalNeuralNetwork,
            'nn': FeedForwardNeuralNetwork,
            'rf': RandomForest
        }

        if algorithm not in models_map:
            raise ValueError(f"Algorithm '{algorithm}' is not supported. Choose from {list(models_map.keys())}")

        self.model = models_map[algorithm]()

    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        if X_test.ndim == 3:
            X_test = np.expand_dims(X_test, axis=0)

        return self.model.predict(X_test)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
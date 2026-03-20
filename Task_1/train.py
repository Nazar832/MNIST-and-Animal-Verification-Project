from tensorflow.keras.datasets import mnist
import os

from classes import MnistClassifier


def prepare_data():
    (X_train, y_train), (_, _) = mnist.load_data()

    X_train = X_train.reshape((60000, 28, 28, 1)).astype('float32') / 255

    return X_train, y_train


def train_and_save_models():
    X_train, y_train = prepare_data()

    algorithms = ['rf', 'nn', 'cnn']

    if not os.path.exists('Task_1/models'):
        os.makedirs('Task_1/models')

    for alg in algorithms:
        clf = MnistClassifier(alg)

        clf.train(X_train, y_train)

        model_path = f"Task_1/models/mnist_{alg}.joblib" if alg == 'rf' else f"Task_1/models/mnist_{alg}.h5"
        clf.save(model_path)


if __name__ == "__main__":
    train_and_save_models()
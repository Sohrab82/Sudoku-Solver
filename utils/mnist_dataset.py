import numpy as np
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


def load_mnist(n_valid=1000):
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train) = shuffle(X_train, y_train)

    X_valid = X_test[:n_valid, :, :]
    X_test = X_test[n_valid:, :, :]
    y_valid = y_test[:n_valid]
    y_test = y_test[n_valid:]

    # convert integers to float; normalise and center the mean
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_valid = X_valid.astype("float32")
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_valid = X_valid / 255.

    # add dimention to data to adapt them to Conv2D input shape
    X_train = tf.expand_dims(X_train, axis=3)
    X_test = tf.expand_dims(X_test, axis=3)
    X_valid = tf.expand_dims(X_valid, axis=3)

    # print data shapes
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_valid.shape)
    print(y_valid.shape)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_test = X_test.shape[0]
    image_shape = X_train.shape[1:]
    n_classes = len(np.unique(y_test))

    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_valid)
    print("Number of testing examples =", n_test)

    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return (X_train, y_train), (X_test, y_test), (X_valid, y_valid), (n_train, n_test, n_valid), n_classes, image_shape

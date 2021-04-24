import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


def incept(x, n1, n3, n5):
    x3 = Conv2D(filters=n3, kernel_size=(3, 3),
                activation='relu', padding='SAME')(x)
    x5 = Conv2D(filters=n5, kernel_size=(5, 5),
                activation='relu', padding='SAME')(x)
    x1 = Conv2D(filters=n1, kernel_size=(1, 1),
                activation='relu', padding='SAME')(x)
    return tf.concat([x3, x5, x1], axis=3)


def model_incept_fun(x, n_classes):
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = incept(x, 16, 64, 32)
    x = MaxPooling2D()(x)

    x = incept(x, 8, 32, 16)
    x = Flatten()(x)
    x = Dense(units=400, activation='relu')(x)
    x = Dense(units=n_classes, activation='softmax')(x)
    return x


def model_lenet_fun(x, n_classes):
    x = Conv2D(filters=6, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(units=120, activation='relu')(x)
    x = Dense(units=84, activation='relu')(x)
    x = Dense(units=n_classes, activation='softmax')(x)
    return x


def load_model(model_fun, image_shape, n_classes, learning_rate, h5_file=None):
    # load model from h5_file with model aarchitecture defined with model_fun function
    input_layer = Input(image_shape)
    x = input_layer
    x = model_fun(x, n_classes=n_classes)
    model = Model(input_layer, x, name='input_layer')
    model.summary()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])
    if h5_file:
        model.load_weights(h5_file)
    return model

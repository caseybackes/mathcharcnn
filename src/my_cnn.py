import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
import pickle
import numpy as np


def build_model(num_categories, filter_size=(3, 3)):
    # - - - THE SIMPLE CNN MODEL
    model = Sequential()

    # - - - ADD LAYERS TO THE MODEL
    model.add(Conv2D(45, filter_size, input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, filter_size))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(20))

    model.add(Dense(num_categories))
    model.add(Activation("sigmoid"))

    # - - - COMPILE THE MODEL
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":

    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))

    # - - - Scale the X data
    X = X / 255
    filter_size = (3, 3)

    # - - - BUILD THE MODEL FROM FUNCTION ABOVE
    model = build_model(filter_size=filter_size)

    # - - - FIT THE MODEL
    model.fit(X, y, batch_size=10, epochs=3, validation_split=0.1)

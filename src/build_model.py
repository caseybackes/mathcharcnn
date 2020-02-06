# for the build_model() f'n 
import tensorflow as tf 
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model # to load a previously trained model (might not be used in this script ...?)



def build_model(X,num_categories,filter_size=(3,3)):
    # - - - THE SIMPLE CNN MODEL
    model = Sequential()

    # - - - ADD LAYERS TO THE MODEL
    model.add(Conv2D(45,filter_size, input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(50,filter_size ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Dropout(.05))

    model.add(Conv2D(50,filter_size ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Dropout(.05))

    model.add(Conv2D(50,filter_size ))
    model.add(Activation('relu'))
    
    model.add(Dropout(.10))

    model.add(Flatten())
    model.add(Dense(20))

    model.add(Dense(10))
    model.add(Activation('relu'))

    # - - - OUTPUT LAYER
    model.add(Dense(num_categories))
    model.add(Activation('softmax'))

    # - - - COMPILE THE MODEL
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.categorical_accuracy, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

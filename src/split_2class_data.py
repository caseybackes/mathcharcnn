# for the create_training_data() f'n
import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
import random
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# for the build_model() f'n 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np




def class_count(list_of_class_strings, data_directory,verbose_only=False):
    class_dict = dict()
    for class_item in list_of_class_strings:
        path = os.path.join(data_directory, class_item) 
        class_dict[class_item] = len(os.listdir(path))
    if verbose_only:
        print(class_dict)
    else:
        return class_dict


def create_training_data(data_directory, categories, shuffle=False, save_pickle=False, holdout_split=.1):
    training_data = []
    for category in categories:
        #print(f"Building class '{category}'")
        path = os.path.join(data_directory, category)
        class_num = categories.index(category) # index the label name to a number 
        #print('class num: ',class_num)
        for img in os.listdir(path):
            #print(os.path.join(data_directory,img))
            im_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) 
            new_array = cv2.resize(im_array, (45,45))
            training_data.append([im_array, class_num])
            #print(f'class: {category},img: ', img)
    print(f"Built {len(categories)} classes.")

    if shuffle:    
        random.shuffle(training_data) # immutable, need not be stored in a var (also memory leak point if you do save to var)

    # - - - Save training data to feature and label sets
    X = []
    y = []

    for features, labels in training_data:
        X.append(features)
        y.append(labels)

    # - - - Reshape as appropriate 
    X = np.array(X).reshape(-1,45,45, 1)
    if len(set(y)) > 2:
        y = to_categorical(y) # for more than one class. 
    
    # - - - DISPLAY THE CLASS DISTRIBUTION
    # class_count(categories,data_directory)
    
    if save_pickle:
        # - - - Pickle save the feature and label datasets
        pickle_out = open("X.pickle", 'wb')
        pickle.dump(X,pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", 'wb')
        pickle.dump(y,pickle_out)
        pickle_out.close()

    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=holdout_split, random_state=42, stratify=y)  

    return X_train, X_test, y_train, y_test 

def build_model(X,num_categories,filter_size=(3,3)):
    # - - - THE SIMPLE CNN MODEL
    model = Sequential()

    # - - - ADD LAYERS TO THE MODEL
    model.add(Conv2D(45,filter_size, input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(20,filter_size ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(10,filter_size ))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Flatten())
    model.add(Dense(10))

    # - - - OUTPUT LAYER
    model.add(Dense(num_categories))
    model.add(Activation('sigmoid'))

    # - - - COMPILE THE MODEL
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

def evaluate_model(X_test,y_test,categories,limit=-1):
    predictions_cat_ix = [np.argmax(row) for row in model.predict(X_test[0:limit])]
    predictions_class_names = [categories[x] for x in predictions_cat_ix]
    y_true_ix = [categories[x] for x in [np.argmax(yi) for yi in y_test[0:limit]]]
    result = np.array([pred==truth for pred,truth in list(zip(predictions_class_names,y_true_ix))])
    acc = result.sum()/len(result)
    return acc
    


if __name__ == "__main__": 

    # - - - Create two classes of data
    categories = ['alpha',
                'beta',
                'pi',
                'sum',
                'v',
                'p',
                'theta']
    data_directory = '/Users/casey/Downloads/handwrittenmathsymbols/extracted_images-1'

    # - - - Run training data collection
    X_train, X_test, y_train, y_test  = create_training_data(data_directory,categories,shuffle=True,save_pickle=False)

    # - - - Scale the X data
    X_train= X_train/255
    X_test = X_test/255
    filter_size = (3,3)
    num_categories = y_train.shape[1] # ex: (6500,[1])

    # - - - BUILD THE MODEL FROM FUNCTION ABOVE
    acc = 0
    while acc < .90:
        model = build_model(X_train,num_categories =num_categories, filter_size=filter_size)

        # - - - FIT THE MODEL
        model.fit(X_train,y_train,batch_size= 100, epochs = 8, validation_split = .1 )

        acc = evaluate_model(X_test,y_test, categories, limit = -1)
        print(f'{"-"*80}\nAccuracy on holdout set: {round(acc,4)}\nClasses:\n{class_count(categories,data_directory)}')


    '''
    REFERENCE: 

    For reading in a pickeled file:
    >>> pickle_in = open("X.pickle", 'rb')
    >>> X = pickle.load(pickle_in)

    NOTES: 
    1/31/2020
        -Successfully split the data from 'alpha' and 'beta' into 
        trainable datasets compatible with Keras' Sequential CNN model. 
        -Successfully trained the model: validation loss ~1e-4 with 100% accuracy
        which is obviously highly suspect. 
         
    2/1/2020
        -The simple CNN is getting 100% accuracy on the 'alpha','beta',and 'pi' images.
        I'm thinking I need to also have a holdout set to predict on ...obviously... and
        get the validation metrics from that dataset. 
        -The accuracy of predictions on the training data with 2 and 3 classes 
        is easily hitting 100% within the first or second (of 3) epocs of training. 
        The holdout set also predicts 100% across 2 and three classes (alpha, beta, pi)
        -I added a hidden layer and am getting ~89% accuracy with six classes 
        at around 1500-2500 samples each (which are split for train and holdout)
            Built 7 classes.
            Train on 13466 samples, validate on 1497 samples
            Epoch 1/5
            13466/13466 [==============================] - 20s 1ms/sample - loss: 1.3504 - accuracy: 0.4929 - val_loss: 0.6974 - val_accuracy: 0.7916
            Epoch 2/5
            13466/13466 [==============================] - 20s 1ms/sample - loss: 0.4839 - accuracy: 0.8507 - val_loss: 0.3947 - val_accuracy: 0.8858
            Epoch 3/5
            13466/13466 [==============================] - 22s 2ms/sample - loss: 0.2914 - accuracy: 0.9118 - val_loss: 0.2752 - val_accuracy: 0.9225
            Epoch 4/5
            13466/13466 [==============================] - 20s 1ms/sample - loss: 0.1997 - accuracy: 0.9413 - val_loss: 0.2046 - val_accuracy: 0.9405
            Epoch 5/5
            13466/13466 [==============================] - 20s 1ms/sample - loss: 0.1522 - accuracy: 0.9534 - val_loss: 0.1665 - val_accuracy: 0.9486
            --------------------------------------------------------------------------------
            Accuracy on holdout set: 0.9633
            Classes:
            {'alpha': 2546, 'beta': 2025, 'pi': 2332, 'sum': 2689, 'v': 1558, 'p': 2680, 'theta': 2796}
            
    
    '''


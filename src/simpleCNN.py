# for the create_training_data() f'n
import numpy as np
from create_training_data import create_training_data
from build_model import build_model
from save_model import save_model
from evaluate_model import evaluate_model


# for the build_model() f'n 
# import tensorflow as tf 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
# import pickle
# from tensorflow.keras.models import load_model # to load a previously trained model (might not be used in this script ...?)

# for save model
from datetime import datetime as dt 


# def class_count(list_of_class_strings, data_directory,verbose_only=False):
#     class_dict = dict()
#     for class_item in list_of_class_strings:
#         path = os.path.join(data_directory, class_item) 
#         class_dict[class_item] = len(os.listdir(path))
#     if verbose_only:
#         print(class_dict)
#     else:
#         return class_dict


# def create_training_data(data_directory, categories, shuffle=False, save_pickle=False, holdout_split=.1):
#     training_data = []
#     print('Building classes...')
#     for category in categories:
#         #print(f"Building class '{category}'")
#         path = os.path.join(data_directory, category)
#         class_num = categories.index(category) # index the label name to a number 
#         #print('class num: ',class_num)
#         for img in os.listdir(path):
#             #print(os.path.join(data_directory,img))
#             im_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) 
#             new_array = cv2.resize(im_array, (45,45))
#             training_data.append([im_array, class_num])
#             #print(f'class: {category},img: ', img)
#     print(f"Built {len(categories)} classes.")

#     if shuffle:    
#         random.shuffle(training_data) # immutable, need not be stored in a var (also memory leak point if you do save to var)

#     # - - - Save training data to feature and label sets
#     X = []
#     y = []

#     for features, labels in training_data:
#         X.append(features)
#         y.append(labels)

#     # - - - Reshape as appropriate 
#     X = np.array(X).reshape(-1,45,45, 1)
#     if len(set(y)) > 2:
#         y = to_categorical(y) # for more than one class. 
    
#     # - - - DISPLAY THE CLASS DISTRIBUTION
#     # class_count(categories,data_directory)
    
#     if save_pickle:
#         # - - - Pickle save the feature and label datasets
#         pickle_out = open("X.pickle", 'wb')
#         pickle.dump(X,pickle_out)
#         pickle_out.close()

#         pickle_out = open("y.pickle", 'wb')
#         pickle.dump(y,pickle_out)
#         pickle_out.close()

#     X_train, X_test, y_train, y_test = train_test_split( 
#         X, y, test_size=holdout_split, random_state=42, stratify=y)  

#     return X_train, X_test, y_train, y_test 

# def build_model(X,num_categories,filter_size=(3,3)):
#     # - - - THE SIMPLE CNN MODEL
#     model = Sequential()

#     # - - - ADD LAYERS TO THE MODEL
#     model.add(Conv2D(45,filter_size, input_shape = X.shape[1:]))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))

#     model.add(Conv2D(50,filter_size ))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))

#     model.add(Conv2D(50,filter_size ))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))

#     model.add(Conv2D(50,filter_size ))
#     model.add(Activation('relu'))

#     model.add(Flatten())
#     model.add(Dense(20))

#     model.add(Dense(10))
#     model.add(Activation('relu'))

#     # - - - OUTPUT LAYER
#     model.add(Dense(num_categories))
#     model.add(Activation('sigmoid'))

#     # - - - COMPILE THE MODEL
#     model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy',tf.keras.metrics.categorical_accuracy])
#     return model

# def evaluate_model(X_test,y_test,categories,model,limit=-1):
#     predictions_cat_ix = [np.argmax(row) for row in model.predict(X_test[0:limit])]
#     predictions_class_names = [categories[x] for x in predictions_cat_ix]
#     y_true_ix = [categories[x] for x in [np.argmax(yi) for yi in y_test[0:limit]]]
#     result = np.array([pred==truth for pred,truth in list(zip(predictions_class_names,y_true_ix))])
#     acc = result.sum()/len(result)
#     return acc


if __name__ == "__main__": 

    # - - - Create two classes of data
    categories = ['alpha',
                'beta',
                'pi',
                'phi',
                'gamma',
                'sum',
                'v',
                'p',
                'theta',
                'times']

    # data_directory = '../data/handwrittenmathsymbols/image-files'
    data_directory = '../../../../DataScienceProjects/handwrittenmathsymbols/extracted_images'

    # - - - Run training data collection
    X_train, X_test, y_train, y_test  = create_training_data(data_directory,categories,shuffle=True,save_pickle=False,scale=True,distribution_plot=True)

    # # - - - Scale the X data # !TODO: [COMPLETE] moved to the create_training_data function
    # X_train= X_train/255
    # X_test = X_test/255

    # - - - HYPERPARAMETERS
    filter_size = (3,3)
    num_categories = y_train.shape[1] # ex: (6500,[1])

    # - - - BUILD THE MODEL AND TRAIN
    '''
    The model tends to crash sometimes. The loss value can turn to 
    NaN at seemingly random times, in which case the validation
    accuracy for that epoch and all epocs after will be ~15%. 
    If this happens, we'll retrain the model until it comes out 
    with better than 90% accuracy'''
    acc = 0
    val_split = .1
    while acc < .90:
        print("Building model...")
        model = build_model(X_train,num_categories =num_categories, filter_size=filter_size)
        print("Fiting training data to model...")
        model.fit(X_train,y_train,batch_size= 64, epochs = 3, validation_split = val_split)
        # - - - EVALUATE MODEL FOR ACCURACY AGAINST HOLDOUT SET
        print('Evaluating trained model against holdout dataset...')
        acc,result = evaluate_model(X_test,y_test, categories, model,limit = -1, return_prediction_array=True)
        print(f'{"-"*80}\nAccuracy on holdout set: {round(acc,6)}')
    
    # - - - SAVE THE MODEL (OPTIONAL)
    saveme= input('Save model? (y/n):  ')
    if 'y' in saveme:
        save_model(model)
        # today = str(dt.now().date())
        # timestamp = str(dt.now().date()) + "T:"+ str(dt.now().time())[0:8] 
        # model.save(f'../models/simpleCNN-{timestamp}.h5')  # creates a HDF5 file 'my_model.h5'
        # print(f"Saved as models/simpleCNN-{timestamp}.h5")
        # with open(f'../models/reports/simpleCNN-{timestamp}.txt','w') as f:
        #     f.write(f'Classes in data: {categories}\n')
        #     f.write(f'Train-to-Holdout Ratio: {1-val_split}\n')
        #     f.write(f'Holdout Accuracy: {acc}\n')
        #     f.close()
        #     print(f"Report of model saved at models/reports/simpleCNN-{timestamp}.txt ")

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


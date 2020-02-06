# for the create_training_data() f'n
import numpy as np
from create_training_data import create_training_data
from build_model import build_model
from save_model import save_model
from evaluate_model import evaluate_model
from datetime import datetime as dt 

if __name__ == "__main__": 

    # - - - DECLARE CLASSES FOR MODEL TRAINING AND PREDICTION
    # categories = ['alpha',
    #             'beta',
    #             'pi',
    #             'phi',
    #             'gamma',
    #             'sum',
    #             'v',
    #             'p',
    #             'theta',
    #             'times']
    # categories_raw = open('categories_all.txt').readlines()
    # categories_raw = open('categories_greek.txt').readlines()
    categories_raw = open('categories_math.txt').readlines()
    categories = [x.split('\n')[0] for x in categories_raw]

    data_directory = '../../../../DataScienceProjects/handwrittenmathsymbols/extracted_images'

    # - - - CREATE TRAINING DATA
    X_train, X_test, y_train, y_test  = create_training_data(data_directory,categories,shuffle=True,save_pickle=False,scale=True,distribution_plot=True)

    # - - - HYPERPARAMETERS
    filter_size = (3,3)
    num_categories = y_train.shape[1] # ex: (6500,[1])
    training_epochs = 5
    batch_size = 256
    val_split = .1

    # - - - BUILD AND TRAIN THE MODEL 
    model = build_model(X_train,num_categories =num_categories, filter_size=filter_size)
    model.class_names = categories # for use later in EDA
    hist = model.fit(X_train,y_train,batch_size= batch_size, epochs = training_epochs, validation_split = val_split)
    acc,result = evaluate_model(X_test,y_test, categories, model,limit = -1, return_prediction_array=True)
    print(f'{"-"*80}\nPerformance on holdout set of {len(y_test)} images: \nAccuracy:{round(acc,6)}\nHistory: {hist.history}')

    # - - - SAVE THE MODEL (OPTIONAL)
    saveme= input('Save model? (y/n):  ')
    if 'y' in saveme:
        save_model(model,categories,val_split, acc, hist)


    # - - - WHERE DOES THE MODEL GO WRONG? WHAT CHARS CAN WE CONSISTANTLY RECOGNIZE? 
    yhat_probs = model.predict(X_test)
    cats= np.array(categories)
    top_3_pred = []
    bad_pred = []
    for i,y in enumerate(yhat_probs):
        true_label = cats[np.argmax(y_test[i])]
        idx = np.argsort(y)[::-1]
        top_cats = cats[idx]
        # print(y)
        #print('TRUE LABEL: ',true_label)
        # print(idx)
        #print('TOP 3 CLASSES: ', top_cats[:3])
        #print('TOP 3 PROBS:   ',y[idx][:3],'\n')
        if top_cats[0] != true_label:
            print('\nBad prediction: ')
            print(y)
            print('TRUE LABEL: ',true_label)
            print(idx)
            print('TOP 3 CLASSES: ', top_cats[:3])
            print('TOP 3 PROBS:   ',y[idx][:3],'\n')
            bad_pred.append([true_label,top_cats[:3],y[idx][:3]])
        top_3_pred.append([true_label,top_cats[:3],y[idx][:3]])
    
    # - - - OF THE BAD PREDICTIONS, WHICH HAD THE CORRECT GUESS AS CHOICE NUMBER 2? 
    second_guess_correct = 0 
    for pred in np.array(bad_pred):
        print(pred)
        print('\n')
        predicted_label = pred[0]
        top3guesses = pred[1]
        top3probs = pred[2]
        if predicted_label in top3guesses:
            second_guess_correct+=1
    #>>> second_guess_correct
    #>>>     All of them

    # - - - What are the common misclassifications? ie: 'pi' is commonly mistaken for 'v' 
    # - - - What is the probability difference between the false classification and the true label? 

    # scp -i ~/.ssh/dsi_nlp_group_keypair.pem -r extracted_images/ ubuntu@ec2-35-163-141-129.us-west-2.compute.amazonaws.com
    # dsi_nlp_group_keypair.pem # has a t2.large with 100gb of block storage
    

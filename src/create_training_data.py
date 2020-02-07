import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime as dt
import random
from class_count import class_count
from eda import plot_class_distribution


def create_training_data(
    data_directory,
    categories,
    shuffle=False,
    save_pickle=False,
    holdout_split=0.1,
    scale=True,
    distribution_plot=False,
):
    training_data = []
    print("Building classes...")
    for category in categories:
        print("Building class", category)
        path = os.path.join(data_directory, category)
        class_num = categories.index(category)  # index the label name to a number
        for img in os.listdir(path):
            if img != ".DS_Store": # <-- where the hell does this keep coming from?!
                im_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(im_array, (45, 45))
                training_data.append([im_array, class_num])
    print(f"Built {len(categories)} classes.")

    if shuffle:
        random.shuffle(
            training_data
        )  # immutable, need not be stored in a var (also memory leak point if you do save to var)

    # - - - Save training data to feature and label sets
    X = []
    y = []

    for features, labels in training_data:
        X.append(features)
        y.append(labels)

    # - - - Reshape as appropriate
    X = np.array(X).reshape(-1, 45, 45, 1)
    if len(set(y)) > 2: # one-hot encodeing for more than one class -> y_i = [0,0,0,1,0,0,0]
        y = to_categorical(
            y
        )  
    # - - - SCALE DATA
    if scale:
        X = X / 255

    # - - - DISPLAY THE CLASS DISTRIBUTION
    if distribution_plot:
        ax = plot_class_distribution(categories)

    if save_pickle:
        # - - - Differentiate model iterations based on timestamp
        tstmp = str(dt.now()).split(".")[0].replace(" ", "-")
        outstring = "../data/X" + tstmp + ".pickle"
        pickle_out = open(outstring, "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        outstring_y = "../data/y" + tstmp + ".pickle"
        pickle_out = open(outstring_y, "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=holdout_split, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

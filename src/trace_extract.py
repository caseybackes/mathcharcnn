import pickle
import matplotlib.pyplot as plt
import numpy as np 

train_pkl = open('../data/train-crohme.pickle','rb')
train = pickle.load(train_pkl,encoding='utf-8', errors='strict')
train_pkl.close()

test_pkl = open('../data/test-crohme.pickle','rb')
test = pickle.load(test_pkl,encoding='utf-8', errors='strict')
test_pkl.close()

classes = open('../data/classes/classes.txt', 'r').read().split()

'''
IMAGES ARE 50x50 pixels
>>> train[0]
{'features': array([1, 1, 1, ..., 1, 1, 1], dtype=uint8),
 'label': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       dtype=int8)}
'''

X_train = []
y_train = []
y_train_labels = []

X_test = []
y_test = []
y_test_labels = []

for x in train:
    X_train.append(x['features'].reshape((50,50)))
    y_train.append(x['label']) # one-hot array
    y_train_labels.append(classes[np.argmax(x['label'])]) # class as name string
for y in test:
    X_test.append(y['features'].reshape((50,50)))
    y_test.append(y['label']) #one-hot array
    y_test_labels.append(classes[np.argmax(y['label'])]) # class name as string

u = ''
while u != 'n':
    for i in range(len(X_train)):
        plt.imshow(X_train[i])
        class_label=y_train_labels[i]
        plt.title(str("Class: " + class_label))
        plt.show(block=False)
        u = input('Continue? ("n" to stop)')
        if u == 'n':
            break




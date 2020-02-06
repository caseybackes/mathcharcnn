import os
import numpy as np 
import matplotlib.pyplot as plt 
from load_model import load_model
from class_count import class_count



def plot_class_distribution(class_list): 
    '''Plots the distribution of classes given a class list
    Returns a plot object.  
    '''
    data_directory = '../../../../DataScienceProjects/handwrittenmathsymbols/extracted_images'
    # all_classes = [os.listdir(os.path.join(data_directory,c)) for c in class_list]
    all_classes = class_list
    print('Completed up to "all classes"')
    # all_classes.remove('.DS_Store')
    classes = class_count(all_classes,data_directory, verbose_only=False) # all classes(folders) in the image-files directory
    print("returned from class_count(): ",classes)

    # - - - WHAT IS THE DISTRIBUTION OF CLASSES? 
    labels = np.array(list(classes.keys()))
    data = np.array(list(classes.values()))
    # sort by most common
    argsorted = data.argsort()[::-1]#[:-1] 
    labels = labels[argsorted]
    data = data[argsorted]

    # --- the data
    y = np.arange(len(data))
    # --- plot
    width = .8
    fig, ax = plt.subplots(figsize=(8, 10)) 
    ax.barh(y, data, width, color='blue', align='center')
    # --- tidy-up and save 
    ax.set_yticks(y) 
    ax.set_yticklabels(labels) 
    ax.xaxis.grid(True) 
    ax.set_ylabel('Class Name'); ax.set_xlabel('Count') 
    ax.set_title("Mathematical Character Histogram") 
    plt.ylim(min(y)-1, max(y)+1)
    fig.tight_layout(pad=0)
    fig.savefig('../plots/Mathematical Character Histogram.png', dpi=125)
    plt.show(block=False)
    return ax
# ---------------------------------------------------------

# - - - WHERE DOES THE MODEL GO WRONG? WHAT CHARS CAN WE CONSISTANTLY RECOGNIZE? 

# model = load_model('../models/simpleCNN-2020-02-04T:13:14:02.h5')
model_id = 'simpleCNN-2020-02-05T:14:12:03'#.h5'
model = load_model(f'../models/{model_id}.h5')

with open(f'../models/reports/{model_id}.txt', 'r') as f:
    #categories = f.readlines()
    categories = f.readline()[17:].rstrip("\n").strip("][").split(', ')
    categories = [x.strip("'") for x in categories]
    # res = categories.strip('][').split(', ') 
    # res.remove('\n')
    f.close()


yhat_probs = model.predict(X_test)
cats= np.array(categories)
top_3_pred = []
bad_pred = []
good_pred = []
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
    else:
        good_pred.append([true_label,top_cats[:3],y[idx][:3]])
    top_3_pred.append([true_label,top_cats[:3],y[idx][:3]])

# - - - OF THE BAD PREDICTIONS, WHICH HAD THE CORRECT GUESS AS CHOICE NUMBER 2? 
second_guess_correct = 0 
for num,pred in enumerate(np.array(bad_pred)):
    print(pred)
    print('\n')
    predicted_label = pred[0]
    top3guesses = pred[1]
    top3probs = pred[2]
    if predicted_label in top3guesses:
        second_guess_correct+=1

#>>> print(second_guess_correct)
#>>> ...All of them, actually. 




# - - - PLOT: PERCENTAGE OF CORRECT CLASSIFICATION PER CLASS
test_vs_prediction = []
y_test_classes_idx = [np.argmax(y) for y in y_test]
y_test_classes = [cats[x] for x in y_test_classes_idx]
yhat_idx = [np.argmax(y) for y in yhat_probs]
yhat_classes = [cats[x] for x in yhat_idx]
for n in range(len(yhat_classes)):
    test_vs_prediction.append([y_test_classes[n], yhat_classes[n]])
test_vs_prediction= np.array(test_vs_prediction)
correct_counts_by_class = [0]*len(categories)
total_samples_in_each_class =[0]*len(categories)
for row in test_vs_prediction:
    total_samples_in_each_class[categories.index(row[0])]+=1
    if row[0] == row[1]: # correct prediction
        true_label = row[0]
        prediction = row[1]
        correct_counts_by_class[categories.index(true_label)]+=1
percentage_correct_by_class = [i/j for i,j in list(zip(correct_counts_by_class,total_samples_in_each_class))]

# VISUALIZE THE ABOVE QUESTION/ANSWER
labels = cats                               #['alpha, beta, pi, phi, gamma, sum, v, p, theta, times'] 
data = np.array(percentage_correct_by_class )
idx = np.argsort(data)[::-1]
data_sorted = data[idx]
labels_sorted = labels[idx]

y = np.arange(len(data))
# --- plot
width = 0.8
fig, ax = plt.subplots(figsize=(8, 3.5)) 
ax.barh(y, data_sorted, width, color='wheat', align='center')
# --- tidy-up and save 
ax.set_yticks(y) 
ax.set_yticklabels(labels_sorted) 
ax.xaxis.grid(True) 
ax.set_ylabel('Classes Trained in this Model'); ax.set_xlabel('Percentage Correctly Classified') 
ax.set_title("Percentage of Correct Classifications By Class Type") 
fig.tight_layout(pad=1) 
plt.show()
# fig.savefig('percent.png', dpi=125) 
# plt.close('all')

# - - - SOME ARE BEING CLASSIFIED INCORRECTLY A LOT. WHICH ONES, AND HOW MANY SAMPLES EXIST FOR IT TO TRAIN THE NETWORK ON ? 
h = np.array(bad_pred)
for h_i in h: 
    char_class = h_i[0] 
    first_guess = h_i[1][0] 
    first_guess_p = h_i[2][0]; first_guess_p = round(first_guess_p,4)
    second_guess = h_i[1][1] 
    second_guess_p = h_i[2][1];second_guess_p = round(second_guess_p,4)
    third_guess = h_i[1][2] 
    third_guess_p = h_i[2][2]; third_guess_p = round(third_guess_p,4)
    
    if char_class == second_guess: 
        print("it was the second guess") 
        print(f"char class: '{char_class}' \t first guess: '{first_guess}' @ {first_guess_p} \tsecond guess: '{second_guess}' @ {second_guess_p}") 
        print('\n') 


# - - - DICTIONARY OF BAD PREDICTION FREQUENCY BY CLASS
bad_pred_count_dict = dict()
for h_i in h[:,0]: 
    if h_i in bad_pred_count_dict.keys(): 
        bad_pred_count_dict[h_i]+=1 
    else: 
        bad_pred_count_dict[h_i] = 1 


labels = np.array([x for x in bad_pred_count_dict.keys() ])
data = np.array([x for x in bad_pred_count_dict.values() ])
idx = np.argsort(data)[::-1]
data_sorted = data[idx]
labels_sorted = labels[idx]

y = np.arange(len(data))
# --- plot
width = 0.8
fig, ax = plt.subplots(figsize=(8, 3.5)) 
# frequency by absolute value of counts
ax.barh(y, data_sorted, width, color='wheat', align='center')
# --- tidy-up and save 
ax.set_yticks(y) 
ax.set_yticklabels(labels_sorted) 
ax.xaxis.grid(True) 
ax.set_ylabel('Classes Trained in this Model'); ax.set_xlabel('Misclassification Frequency') 
ax.set_title("Counts of Misclassifications by Character Class") 
fig.tight_layout(pad=1) 
plt.show()

# how many of each class are in the test set? 
y_test_str = [categories[np.argmax(yy)] for yy in y_test]
y_test_by_count = dict()
for item in y_test_str:
    if item in y_test_by_count.keys():
        y_test_by_count[item]+=1
    else:
        y_test_by_count[item]=1

data_sorted_normed = dict()
zipped_sorted_labels_data  = [(x,y) for x,y in list(zip(labels_sorted, data_sorted))]
for ld in zipped_sorted_labels_data:
    c,freq = ld
    freq_normed = freq/y_test_by_count[c]
    data_sorted_normed[c] = freq_normed
y = np.arange(len(data))
# --- plot
width = 0.8
fig, ax = plt.subplots(figsize=(8, 3.5)) 
# frequency by absolute value of counts
ax.barh(y, data_sorted_normed.values(), width, color='wheat', align='center')
# --- tidy-up and save 
ax.set_yticks(y) 
ax.set_yticklabels(labels_sorted) 
ax.xaxis.grid(True) 
ax.set_ylabel('Classes Trained in this Model'); ax.set_xlabel('Misclassification Frequency Normalized') 
ax.set_title("Normalized Counts of Misclassifications by Character Class") 
fig.tight_layout(pad=1) 
plt.show()

# - - - What are the common misclassifications? ie: 'pi' is commonly mistaken for 'v' 
# - - - What is the probability difference between the false classification and the true label? 



'''
[('rightarrow', ':', 1703, 'images'),
 ('sin', ':', 4293, 'images'),
 ('i', ':', 5140, 'images'),
 ('R', ':', 2671, 'images'),
 ('0', ':', 6914, 'images'),
 ('times', ':', 3251, 'images'),
 ('leq', ':', 973, 'images'),
 ('9', ':', 3737, 'images'),
 ('gt', ':', 258, 'images'),
 ('+', ':', 25112, 'images'),
 ('u', ':', 1269, 'images'),
 ('N', ':', 10862, 'images'),
 ('mu', ':', 177, 'images'),
 ('exists', ':', 21, 'images'),
 ('gamma', ':', 409, 'images'),
 ('Delta', ':', 137, 'images'),
 ('div', ':', 868, 'images'),
 ('infty', ':', 1783, 'images'),
 ('G', ':', 1692, 'images'),
 ('beta', ':', 2025, 'images'),
 ('in', ':', 47, 'images'),
 (',', ':', 1508, 'images'),
 ('7', ':', 2909, 'images'),
 ('forward_slash', ':', 199, 'images'),
 ('{', ':', 376, 'images'),
 ('=', ':', 13104)
 ('pm', ':', 802, 'images'), 
 ('cos', ':', 2986, 'images')]
 '''

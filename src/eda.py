import os,cv2
import numpy as np 
import matplotlib.pyplot as plt 
from load_model import load_model
from class_count import class_count
from evaluate_model import evaluate_model



def class_prob_dist(y_test,yhat_probs,y_test_str,categories):
    count_dict = dict()
    rows = len(y_test)
    for i in range(rows):
        #its an alpha
        clss= y_test_str[i]
        # what was the top three probs and 
        idx_top3 = np.argsort(yhat_probs[i])[::-1][0:3]
        top3probs = [yhat_probs[i][x] for x in idx_top3]
        top3pred = [categories[x] for x in idx_top3]
        if clss not in count_dict.keys():
            count_dict[clss] = [0,0,0]
        for i in range(3):
            if clss == top3pred[i]:
                count_dict[clss][i]+=1
    return count_dict 


def add_value_labels(ax,sigfigs,spacing=5, ):
    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space = spacing #*= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = round(x_value,sigfigs)#"{:.4f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(space, 0),          # Horizontally shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            va='center',                # Vertically center label
            ha=ha)                      # Horizontally align label differently for
                                        # positive and negative values.


def plot_class_distribution(class_list): 
    '''Plots the distribution of classes given a class list
    Returns a plot object.  
    '''
    data_directory = '../../../../DataScienceProjects/handwrittenmathsymbols/extracted_images'
    # all_classes = [os.listdir(os.path.join(data_directory,c)) for c in class_list]
    all_classes = class_list
    # all_classes.remove('.DS_Store')
    classes = class_count(all_classes,data_directory, verbose_only=False) # all classes(folders) in the image-files directory
    # print("returned from class_count(): ",classes)

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
    add_value_labels(ax,sigfigs=0)
    fig.tight_layout(pad=0)
    # fig.savefig('../plots/Mathematical Character Histogram.png', dpi=125)
    # plt.show(block=False)
    return ax

# ---------------------------------------------------------

# - - - WHERE DOES THE MODEL GO WRONG? WHAT CHARS CAN WE CONSISTANTLY RECOGNIZE? 

# model = load_model('../models/simpleCNN-2020-02-04T:13:14:02.h5')
model_id = 'simpleCNN-2020-02-05T:20:53:54'#.h5'
model = load_model(f'../models/{model_id}.h5')

with open(f'../models/reports/{model_id}.txt', 'r') as f:
    #categories = f.readlines()
    categories = f.readline()[17:].rstrip("\n").strip("][").split(', ')
    categories = [x.strip("'") for x in categories]
    # res = categories.strip('][').split(', ') 
    # res.remove('\n')
    f.close()
ax = plot_class_distribution(categories)


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
        # print('\nBad prediction: ')
        # print(y)
        # print('TRUE LABEL: ',true_label)
        # print(idx)
        # print('TOP 3 CLASSES: ', top_cats[:3])
        # print('TOP 3 PROBS:   ',y[idx][:3],'\n')
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
print("Of the ",len(np.array(bad_pred)), 'bad predictions, ',second_guess_correct, 'had the correct prediction as the second highest probability of classification.')
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
fig, ax = plt.subplots(figsize=(8, 10)) 
ax.barh(y, data_sorted, width, color='blue', align='center')
# --- tidy-up and save 
ax.set_yticks(y) 
ax.set_yticklabels(labels_sorted) 
ax.xaxis.grid(True) 
ax.set_ylabel('Classes Trained in this Model'); ax.set_xlabel('Percentage Correctly Classified') 
ax.set_title("Percentage of Correct Classifications By Class Type") 
add_value_labels(ax,3)
fig.tight_layout(pad=1) 
plt.show()
# fig.savefig('percent.png', dpi=125) 
# plt.close('all')

# - - - SOME ARE BEING CLASSIFIED INCORRECTLY A LOT. WHICH ONES, AND HOW MANY SAMPLES EXIST FOR IT TO TRAIN THE NETWORK ON ? 
# h = np.array(bad_pred)
# for h_i in h: 
#     char_class = h_i[0] 
#     first_guess = h_i[1][0] 
#     first_guess_p = h_i[2][0]
#     first_guess_p = round(first_guess_p,4)
#     second_guess = h_i[1][1] 
#     second_guess_p = h_i[2][1]
#     second_guess_p = round(second_guess_p,4)
#     third_guess = h_i[1][2] 
#     third_guess_p = h_i[2][2]
#     third_guess_p = round(third_guess_p,4)
    
#     if char_class == second_guess: 
#         print("it was the second guess") 
#         print(f"char class: '{char_class}' \t first guess: '{first_guess}' @ {round(first_guess_p,4)}% \tsecond guess: '{second_guess}' @ {second_guess_p}%") 
#         print('\n') 


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
fig, ax = plt.subplots(figsize=(8,10)) 
# frequency by absolute value of counts
ax.barh(y, data_sorted, width, color='blue', align='center')
# --- tidy-up and save 
ax.set_yticks(y) 
ax.set_yticklabels(labels_sorted) 
ax.xaxis.grid(True) 
ax.set_ylabel('Classes Trained in this Model'); ax.set_xlabel('Misclassification Frequency') 
ax.set_title("Counts of Misclassifications by Character Class") 
add_value_labels(ax,0)
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
fig, ax = plt.subplots(figsize=(8, 10)) 
# frequency by absolute value of counts
ax.barh(y, data_sorted_normed.values(), width, color='blue', align='center')
# --- tidy-up and save 
ax.set_yticks(y) 
ax.set_yticklabels(labels_sorted) 
ax.xaxis.grid(True) 
ax.set_ylabel('Classes Trained in this Model'); ax.set_xlabel('Misclassification Frequency Normalized') 
ax.set_title("Normalized Counts of Misclassifications by Character Class") 
add_value_labels(ax,3)
fig.tight_layout(pad=1) 
plt.show()

# - - -  OF THE CLASSES COMMONLY MISCLASSIFIED, SHOW SOME EXAMPLES OF THE IMAGES
fig = plt.figure(figsize=(8,3)) 
# fig.text(x=0.01, y=0.01, s='Figure',color='#888888', ha='left', va='bottom', fontsize=20)
# class_imgs1= os.listdir(os.path.join(data_directory,category)) 
# the five worst image misclassifications
idx_worst_misclass = np.argsort(list( data_sorted_normed.values()))[0:6]
worst_misclassed = [categories[x] for x in idx_worst_misclass]

def plot_imgrows(rows,cols,categories, data_directory,title,imsize=80,):
    fig,axs = plt.subplots(rows,cols,figsize =(rows,cols))
    plt.suptitle(title,fontsize=20)

    for c in range(len(categories)-1):
        for i in range(cols):
            print("c: ", c, '\t i: ', i)
            class_path = os.path.join(data_directory, categories[c])
            pickone = random.choice(os.listdir(class_path))
            path_to_one = os.path.join(class_path, pickone)
            im_array = cv2.imread(path_to_one,cv2.IMREAD_GRAYSCALE)
            im_array_resized = cv2.resize(im_array,(imsize,imsize))
            axs[c, i].imshow(im_array_resized,cmap='gist_gray')
            axs[c, i].set_xticks([]); axs[c,i].set_yticks([])
            # axs[c, 0].set_ylabel(categories[c]).set_rotation(0)
            axs[c, 0].set_title(categories[c])
    # fig.tight_layout(pad=0)
    plt.show()
    return ax 
# plot_imgrows(6, 10, categories,data_directory, 'Sample of Some Classes Trained in CNN Model',imsize=50)
'''
fig.tight_layout(
    renderer=None,
    pad=1.08,
    h_pad=None,
    w_pad=None,
    rect=None,
)
'''
#!TODO: TURN THE FOLLOWING INTO A FUNCTION AND USE FOR THE WORST OFFENDERS AND THE CLASS-SET TRAINED ON. 
# fig, axs = plt.subplots(5, 20,figsize=(15,6))
# for c in range(len(worst_misclassed)-1):
#     for i in range(20):
#         class_path = os.path.join(data_directory, worst_misclassed[c])
#         pickone = random.choice(os.listdir(class_path))
#         path_to_one = os.path.join(class_path,pickone)
#         im_array = cv2.imread(path_to_one,cv2.IMREAD_GRAYSCALE) 
#         im_array_resized = cv2.resize(im_array, (80,80))
#         axs[c, i].imshow(im_array_resized,cmap='gist_gray')
#         axs[c, i].set_xticks([]); axs[c,i].set_yticks([])
#         axs[c, 0].set_ylabel([c])
# fig.text(x=0.31, y=0.01, s='Random samples from the classes most misclassified', \
#     color='#888888', ha='center', va='bottom', fontsize=20)
# plt.suptitle("Top 5 Worst Offenders",fontsize=20)
# plt.show()



# - - - What are the common misclassifications? ie: 'pi' is commonly mistaken for 'v' 
# - - - What is the probability difference between the false classification and the true label? 

# _,result = evaluate_model(X_test, y_test, categories, model, limit=-1, return_prediction_array=True)


count_dict = class_prob_dist(y_test, yhat_probs,y_test_str, categories)

# --- get the data
guess1 = np.array(list(count_dict.values()))[:,0]
guess2 = np.array(list(count_dict.values()))[:,1]
guess3 = np.array(list(count_dict.values()))[:,2]
labels = list(count_dict.keys())

# --- the plot – left then right
fig = plt.figure(figsize=(11,10))
fig.suptitle('Frequency of 1st, 2nd, and 3rd \nHighest Probability for Correct Classification',fontsize=25)

for i in range(len(labels)):
    ax = fig.add_subplot(6,4,i+1)
    ax.bar(np.arange(3), list(count_dict.values())[i], width=.6, color=['green','orange','red'])#,   label=['1','2','3']) 
    ax.set_title(labels[i])
    ax.set_xticklabels(['1','2','3'])
    ax.set_xticks([0,1,2])
fig.tight_layout()






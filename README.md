# Handwritten Mathematical Character Recognition with a Simple CNN
### _AKA: MathChar Wreck-ignition_

## Motivation
In the annual competition known as CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions), handwritten mathematical expression datasets are collected by participating institutions in order to build a larger and open corpus of data for mathematical character and expression recognition. All institutions can evaluate their models on this larger dataset and compare performance. Since the first competition in 2011, each iteration's datasets have been aggregated with any newly incoming data. In this project, I attempt to develop a simple CNN architecture trained to recognize / classify individual characters parsed from full mathematical expressions. 

## Data Description

The CROHME dataset contains handwritten mathematical expressions collected though an online input format known as INKML, which is a XML markup [Ref 1] from a pen-based input for writing various expressions [Ref 2]. 

```
<ink xmlns="http://www.w3.org/2003/InkML">
<traceFormat>
<channel name="X" type="decimal"/>
<channel name="Y" type="decimal"/>
</traceFormat>
<annotation type="writer">w123</annotation>
<annotation type="truth">$a<\frac{b}{c}$</annotation>
<annotationXML type="truth" encoding="Content-MathML">
<math xmlns="http://www.w3.org/1998/Math/MathML">
<mrow>
<mi xml:id="A">a</mi>
<mrow>
<mo xml:id="B"><</mo>
<mfrac xml:id="C">
<mi xml:id="D">b</mi>
<mi xml:id="E">c</mi>
</mfrac>
</mrow>
</mrow>
</math>
</annotationXML>
<trace id="1">985 3317, ..., 1019 3340</trace>
...
<trace id="6">1123 3308, ..., 1127 3365</trace>
<traceGroup xml:id="7">
<annotation type="truth">Ground truth</annotation>
<traceGroup xml:id="8">
<annotation type="truth">a</annotation>
<annotationXML href="A"/>
<traceView traceDataRef="1"/>
<traceView traceDataRef="2"/>
</traceGroup>
...
</traceGroup>
</ink>
```

From the INKML files provided in the CROHME dataset, the traces (pen strokes) for individual characters can be isolated and saved as individual images. This processed dataset was retrieved online to avoid redeveloping an existing script to parse the characters [Ref 3]. Onced parsed, there are 300,000+ images of individual characters. These data can be parsed into a directory structure containing images of each class, totaling 82 classes of characters. 

![fig-dirstruct][dirstruct]

[dirstruct]: plots/directorystructure.png "Class Directory Structure"

***For the purposes of this project***, I will select a subset of this data for training and evaluating a simple CNN model to classifiy. Specifically, the following mathematical characters (which total 125480 samples):
> pm,  infty,  div,  gt,  forward_slash,  leq,  times,  sin,  +,  cos,  -,  sqrt,  lim,  neq,  log,  ldots,  lt,  theta,  prime,  =,  tan,  e,  ),  geq


## [IMAGE OF SAMPLE OF INDIVIDUAL CHARS]

![figsample][sample-div]

[sample-div]: plots/sample_div.png

![figsample][sample-forwardslash]

[sample-forwardslash]: plots/sample_forwardslash.png

![figsample][sample-gt]

[sample-gt]: plots/sample_gt.png

![figsample][sample-infty]

[sample-infty]: plots/sample_infty.png

![figsample][sample-leq]

[sample-leq]: plots/sample_leq.png

![figsample][sample-pm]

[sample-pm]: plots/sample_pm.png

![figsample][sample-sqrt]

[sample-sqrt]: plots/sample_sqrt.png

![figsample][sample-theta]

[sample-theta]: plots/sample_theta.png

![figsample][sample-times]

[sample-times]: plots/sample_times.png


## EDA

The first thing to consider in a categorical classifier is how the class samples are distributed. A simple histogram immediately shows a desparate class imbalance. This is important to keep in mind when training a model - the class imbalance must be addressed when selecting a train-test split. Fortunately, we can allow sklearn to evenly distribute the classes among the training and test sets of images. 

![figclassdist][classdist]

[classdist]: plots/classdistribution.png

On visual inspection of the images, it appears many of the math symbols are EXTREMELY similar. For example, in the sample images above, we notice the 'pm' class samples have nearly the exact same stroke for the minus sign at the bottom of the image. The plus sign varies a little in curvature, but these are very closely related images. Part of the reason these images are together is because of the filename similarity - i.e.: they likely came from the same person. This raises the question of how many different people contributed to the generation of class samples across all samples. Unfortunately, this information is not provided with the data, though it would have been insightful to have it. 

Also noteworthy, there are two very different characters being used for the 'div' class - the traditional line-with-dots as well as the slash. If running all samples through a CNN model, it will be interesting to see how the model responds to two very different characters used for the same mathematical operation. 




## CNN Model
After exploring the common architectures for convolutional neural networks I decided on a simple architecture (described below) that consists of 2d convolutions, max pooling, occasional dropout, and a dense layer that feeds a final output layer with as many nodes as there are classes in the training data. Activation functions are all 'relu' up to the last layer which has a 'softmax' activation. This model uses `categorical_crossentropy` for a loss function, and measures `accuracy`, `precision`, and `recall` provided by `Tensorflow.keras.metrics`. I trained the model for 5 epochs. 


```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 43, 43, 45)        450       
_________________________________________________________________
activation (Activation)      (None, 43, 43, 45)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 41, 41, 50)        20300     
_________________________________________________________________
activation_1 (Activation)    (None, 41, 41, 50)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 20, 20, 50)        0         
_________________________________________________________________
dropout (Dropout)            (None, 20, 20, 50)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 18, 18, 50)        22550     
_________________________________________________________________
activation_2 (Activation)    (None, 18, 18, 50)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 9, 9, 50)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 9, 9, 50)          0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 7, 50)          22550     
_________________________________________________________________
activation_3 (Activation)    (None, 7, 7, 50)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 7, 7, 50)          0         
_________________________________________________________________
flatten (Flatten)            (None, 2450)              0         
_________________________________________________________________
dense (Dense)                (None, 20)                49020     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                210       
_________________________________________________________________
activation_4 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 24)                264       
_________________________________________________________________
activation_5 (Activation)    (None, 24)                0         
=================================================================
Total params: 115,344
Trainable params: 115,344
Non-trainable params: 0
_________________________________________________________________
```


### PRO-TIP: 
> One important takeaway for building a multi-class CNN is that the y-labels need to be one-hot encoded by class.
> 
> `from tensorflow.keras.utils import to_categorical`

```
>>> y_test[0:3]
array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]], dtype=float32)
```

## Model Performance 
While training the model, I observed that it was perfoming fairly well (read as "miraculously I scripted a reporting tool that would record the metrics on the model with the architecture. Below is the tabled version of the report.  


![fig-trainprogress][trainprogress]

[trainprogress]: plots/epoch4of5training.png


#### ***From Model.History()***

| Metric | Epoch 1 | Epoch 2 | Epoch 3| Epoch 4 | Epoch 5 | 
|----|----|----|----|----|----|
|loss | 0.799 | 0.138 | 0.0739 | 0.052 | 0.041|
|accuracy| 0.788| 0.964| 0.980 | 0.986 | 0.989 |
|categorical_accuracy| 0.788| 0.964 | 0.980| 0.986| 0.989|
|precision|0.955 | 0.976| 0.984| 0.988| 0.990|
|recall| 0.713|  0.956| 0.977| 0.984| 0.988|
|val_loss| 0.197| 0.086| 0.060| 0.048|0.061|
|val_accuracy| 0.952| 0.979|0.984| 0.987|0.982|
|val_categorical_accuracy|0.952 | 0.979| 0.984 |0.987| 0.982|
|val_precision| 0.974| 0.984| 0.988|  0.989| 0.983|
|val_recall| 0.939 | 0.975| 0.981| 0.986| 0.981|

We can also take a look at how well the model performed on a 'per class' basis. The immediately obvious observation is that the `prime`, `tan`, `gt`, and `forward_slash` were classified correctly less than 90% of the time in the holdout set. Conversely, `sin`, `lt`, and `geq` were classified correctly every time in the holdout set. 

![fig-percentcorrect][percentcorrect]

[percentcorrect]: plots/percentcorrect.png


## Where does the model go wrong? 
While most of the learning metrics above might initially seem satisfying, the real question is "Where could it be better?"

I took a look at the number of times each class was "misclassified", and found that `tan`, `plus`, `infty`, and `prime` had the highest number of incorrect classification. 

![fig-countmisclass][countmisclass]

[countmisclass]: plots/countsmisclass.png


This raises an obvious question about how the samples are distributed across the classes during training. So we divide each count of misclassification for each class by the number of samples in the test set for that class. This chart actually provides the same information as the "percent correct" bar chart, though frames the information as "this is how often the model predicts incorrectly on the test dataset for this class"

![fig-normmisclass][normmisclass]

[normmisclass]: plots/normedcountsmisclass.png

If these classes are so difficult to predict, it makes me want to take another look at the images for these specific classes. 

![fig-top5worst][top5]

[top5]: plots/top5worst.png


## Conclusion
After seeing the decently successful results for the simple CNN for classifying these characters, it seems there are some classes the model struggles with, and reasonably so. A `prime` character is actually just a very small line, and during processing was resized to the same aspect ratio as characters that are typicall much larger. It might make more sense to classify the `prime` symbols as such depending on the context of where it was in the equation.  It's understandabel 

## Future Work

## References
- Ref 1: CROHME Data File Format. https://www.isical.ac.in/~crohme/data2.html
- Ref 2: CROHME Call for participation. https://www.isical.ac.in/~crohme/crohme14.html
- Ref 3: Isolated Character dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols

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
I scripted a reporting tool that would record the metrics on the model and to 


```

Classes in data: ['pm', 'infty', 'div', 'gt', 'forward_slash', 'leq', 'times', 'sin', '+', 'cos', '-', 'sqrt', 'lim', 'neq', 'log', 'ldots', 'lt', 'theta', 'prime', '=', 'tan', 'e', ')', 'geq']
Model: Sequential
Train-to-Holdout Ratio: 0.9
Holdout Accuracy: 0.9822268271299912
Model.History:
loss: [0.7992878800226556, 0.13865616373838666, 0.07399723947373076, 0.052470665467551345, 0.04136091020918481]
accuracy: [0.788524, 0.9641276, 0.98046005, 0.98649126, 0.98913795]
categorical_accuracy: [0.788524, 0.9641276, 0.98046005, 0.98649126, 0.98913795]
precision: [0.9551199, 0.97675145, 0.98462605, 0.9887253, 0.99080443]
recall: [0.713798, 0.9561089, 0.97733134, 0.98446447, 0.98802614]
val_loss: [0.19726133456462586, 0.08629346356667976, 0.06001814073045681, 0.04824599075078838, 0.0614393260697741]
val_accuracy: [0.9526297, 0.9791039, 0.9841509, 0.9879582, 0.98246855]
val_categorical_accuracy: [0.9526297, 0.9791039, 0.9841509, 0.9879582, 0.98246855]
val_precision: [0.97446495, 0.9849839, 0.988757, 0.98952323, 0.98358476]
val_recall: [0.93934834, 0.97573936, 0.98114043, 0.98680717, 0.9814946]

```



## Where does the model go wrong? 

## Future Work

## References
- Ref 1: CROHME Data File Format. https://www.isical.ac.in/~crohme/data2.html
- Ref 2: CROHME Call for participation. https://www.isical.ac.in/~crohme/crohme14.html
- Ref 3: Isolated Character dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols

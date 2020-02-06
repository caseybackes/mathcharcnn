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



## EDA
The first thing to consider in a categorical classifier is how the class samples are distributed. A simple histogram immediately shows a desparate class imbalance. 

![fig-classdist][class-dist]

[class-dist]: plots/classdistribution.png


## CNN Model

## Model Performance 

## Where does the model go wrong? 

## Future Work

## References
- Ref 1: CROHME Data File Format. https://www.isical.ac.in/~crohme/data2.html
- Ref 2: CROHME Call for participation. https://www.isical.ac.in/~crohme/crohme14.html
- Ref 3: Isolated Character dataset: https://www.kaggle.com/xainano/handwrittenmathsymbols

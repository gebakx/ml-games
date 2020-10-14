class: center, middle

## Artificial Intelligence

# Machine Learning for Games

<br>

Gerard Escudero & Samir Kanaan, 2019

<br>

![:scale 50%](figures/inverse.png)

.footnote[Source: [inverse](https://www.inverse.com/article/31467-artificial-intelligence-computer-human-game) ]

---
class: left, middle, inverse

# Outline

* .cyan[Introduction]

* Machine Learning

* Deep Learning

* Reinforcement Learning

* References

---

# Classification Data Example

.center[
class      | sepal <br> length | sepal <br> width | petal <br> length | petal <br> width 
:--------- | -----: | ----: | -----: | ----: 
setosa     | 5.1    | 3.5   | 1.4    | 0.2
setosa     | 4.9    | 3.0   | 1.4    | 0.2
versicolor | 6.1    | 2.9   | 4.7    | 1.4 
versicolor | 5.6    | 2.9   | 3.6    | 1.3 
virginica  | 7.6    | 3.0   | 6.6    | 2.1 
virginica  | 4.9    | 2.5   | 4.5    | 1.7 
150 rows or examples (50 per class).red[*] 
]

* The .blue[class] or .blue[target] column is usually refered as vector .blue[_Y_] 

* The matrix of the rest of columns (.blue[attributes] or .blue[features]) is usually referred
as matrix .blue[_X_]

.footnote[.red[*] _Source_ : _Iris_ problem UCI repository (Frank &amp; Asunción, 2010)]

---

# Main objective

.large[Build from data a .blue[model] able to give a prediction to new .blue[unseen] examples.]

.center[![model](figures/model.png)]

where:
* _data = previous table_
* _unseen = [4.9, 3.1, 1.5, 0.1]_
* _prediction = "setosa"_

---

# Regression Data Example

.center[
quality | density | pH   | sulphates | alcohol
------- | ------- | ---- | --------- | -------
6       | 0.998   | 3.16 | 0.58      | 9.8
4       | 0.9948  | 3.51 | 0.43      | 11.4
8       | 0.9973  | 3.35 | 0.86      | 12.8
3       | 0.9994  | 3.16 | 0.63      | 8.4
7       | 0.99514 | 3.44 | 0.68      | 10.55
1599 examples & 12 columns (11 attributes + 1 target).red[*]
]

The main diference between classification and regression is the _Y_ or target values:

* .blue[Classification]: discrete or nominal values <br>
Example: _Iris_, {“setosa”, “virginica”, “versicolor”}.

* .blue[Regression]: continuous or real values <br>
Example: _WineQuality_, values from 0 to 10.

.footnote[.red[*] _Source_ : _wine quality_ problem from UCI repository (Frank &amp; Asunción, 2010)]
 
---

# Example method: 

#### kNN with k=1

* How can be give a prediction to next examples?

.center[
| class | sep-len | sep-wid | pet-len | pet-wid |
|:------|:--------|:--------|:--------|:--------|
| ??    | 4.9     | 3.1     | 1.5     | 0.1     |
Unseen classification example on _Iris_]

.center[
| target | density | pH   | sulphates | alcohol |
|:-------|:--------|:-----|:----------|:--------|
| ??     | 0.99546 | 3.29 | 0.54      | 10.1    |
Unseen regression example on _WineQuality_]

* Let’s begin with a representation of the problems...

---

# Classification Data Example

![:scale 90%](figures/knn.png)

---

# Regression Data Example

![:scale 90%](figures/knnR.png)

---

# 1 Nearest Neighbors algorithm

* classification & regression

$$h(T)=y_i$$

.center[where $i = argmin_i(distance(X_i,T))$, $n=\vert features\vert$]
.center[and $distance(X,Z) = \sqrt{(x_1-z_1)^2+\ldots+(x_n-z_n)^2}$]

* Classification example (Iris):
  - distances: [0.47, 0.17, 3.66, 2.53, 6.11, 3.45]
  - prediction = setosa (0.17)

* Regression example (WineQuality):
  - distances: [0.33, 1.32, 2.72, 1.71, 0.49] 
  - prediction = 6 (0.33)

--

* .blue[lazy learning]: it means that the kNN does nothing in learning step
  - it calculates all in classify step

* This can produce some problems real time applications
  - such as .blue[Games]

---

# How about games?

#### Application examples:

- .blue[Classification]: decision about braking a car.

| Brake? | Distance | Speed |
|-------:|---------:|------:|
| Y      | 2.4      | 11.3  |
| Y      | 3.2      | 70.2  |
| N      | 75.7     | 72.7  |
| N      | 2.8      | 15.2  |
| %?     | 79.2     | 12.1  |
.center[Source: (Millington, 2019)]

- .blue[Regression]: required amount of force in a curve.

---

# Decision Learning

- A basic application is learning .blue[decisions] from .blue[observations]

  - _decisions_ will become _classes_ ($Y$)

  - _observations_ will become _features_ ($X$)

- Some kind of measure is needed to evaluate the model
 
- .blue[The Balance of Effort]

  - Many times is harder learning than human dessign (such as Behaviour Tree)

---

# Some categories

#### When?

- .blue[Online]: learn as playing

- .blue[Offline]: 
  - learn from saved matches
  - .blue[bootstrapping]: AI component playing among them <br>
    Same algorithm with diferent paramenters <br>
    Diferent algorithms <br>
    ([See chess example](https://arxiv.org/pdf/1509.01549v2.pdf))

#### What?

- .blue[Intra-Behaviour]: atomic behaviour 
  - Example: previous examples of braking or curves

- .blue[Inter-Behaviour]: learns Decision Taking layer
  - Example: [SC2LE(StarCraft II Learning Environment)](https://arxiv.org/abs/1708.04782)

---

# Action Prediction

- It is a simple technique that tries to guess player movements from previous recordings

- Human behaviour is not random

- It is also called RPS from: rock-paper-scissor game

- Example for a simple RPS:
  - Movement: Left, Right
  - Recording: "LRRLRLLL"

- It requires a window size
  - Example: previous 3 movements

- It can produce unbeatable AIs
  - Some level adjustment would be needed

---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .cyan[Machine Learning]

  - .cyan[Naïve Bayes]

  - Decision Trees

* Deep Learning

* Reinforcement Learning

* References

---

# Maximum likelihood estimation

| class      | cap-shape | cap-color | gill-size | gill-color |
|:-----------|:----------|:----------|:----------|:-----------|
| poisonous  | convex    | brown     | narrow    | black      |
| edible     | convex    | yellow    | broad     | black      |
| edible     | bell      | white     | broad     | brown      |
| poisonous  | convex    | white     | narrow    | brown      |
| edible     | convex    | yellow    | broad     | brown      |
| edible     | bell      | white     | broad     | brown      |
| poisonous  | convex    | white     | narrow    | pink       |
.center[up to 8 124 examples & 22 attributes .red[*]]


- What is $P(poisonous)$?

$$P(poisonous)=\frac{N(poisonous)}{N}=\frac{3}{7}\approx 0.429$$

.footnote[.red[*]  _Source_ : _Mushroom_ problem from UCI repository (Frank &amp; Asunción, 2010)]

---

# Naïve Bayes

#### Learning Model

$$\text{model}=[P(y)\simeq\frac{N(y)}{N},P(x_i|y)\simeq\frac{N(x_i|y)}{N(y)};\forall y \forall x_i]$$

.col5050[
.col1[
| $y$       | $P(y)$ |
|:----------|-------:|
| poisonous | 0.429  |
| edible    | 0.571  |
]
.col2[
| attr:value       | poisonous | edible |
|:-----------------|----------:|-------:|
| cap-shape:convex | 1         | 0.5    |
| cap-shape:bell   | 0         | 0.5    |
| cap-color:brown  | 0.33      | 0      |
| cap-color:yellow | 0         | 0.5    |
| cap-color:white  | 0.67      | 0.5    |
| gill-size:narrow | 1         | 0      |
| gill-size:broad  | 0         | 1      |
| gill-color:black | 0.33      | 0.25   |
| gill-color:brown | 0.33      | 0.75   |
| gill-color:pink  | 0.33      | 0      |
]
]

---

# Naïve Bayes

#### Classification

$$h(T) \approx argmax_y P(y)\cdot P(t_1|y)\cdot\ldots\cdot P(t_n|y)$$

- Test example $T$:

| class | cap-shape | cap-color | gill-size | gill-color |
|:------|:----------|:----------|:----------|:-----------|
| ??    | convex    | brown     | narrow    | black      |

- Numbers:
$$P(poisonous|T) = 0.429 \cdot 1 \cdot 0.33 \cdot 1 \cdot 0.33 = 0.047$$
$$P(edible|T) = 0.571 \cdot 0.5 \cdot 0 \cdot 0 \cdot 0.25 = 0$$
- Prediction: $$h(T) = poisonous$$

---

# Naïve Bayes

#### Notes

- It needs a smoothing technique to avoid zero counts <br>
  - Example: Laplace
$$P(x_i|y)\approx\frac{N(x_i|y)+1}{N(y)+N}$$

- It is empiricaly a decent classifier but a bad estimator
  - This means that $P(y|T)$ is not a good probability 

#### Implementation

- Using LINQ of C#: [view](codes/naiveBayes.html).red[*] / [output](codes/modelNB.txt) / [download code](codes/naiveBayes.cs)

.footnote[.red[*] Formated with http://hilite.me/]

---

# Gaussian Naïve Bayes I

* How about numerical features?

| Brake? | Distance | Speed |
|-------:|---------:|------:|
| Y      | 2.4      | 11.3  |
| Y      | 3.2      | 70.2  |
| N      | 75.7     | 72.7  |
| N      | 2.8      | 15.2  |
| %?     | 79.2     | 12.1  |
.center[Source: (Millington, 2019)]

$$P(x_i|y)=\frac{1}{\sqrt{2\pi\sigma_y^2}}\exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}\right)$$

.center[where $\mu_y=\frac{x_1^y+\cdots+x_n^y}{n_y}$ and $\sigma_y^2=\frac{(x_1^y - \mu_y)^2+\cdots+(x_n^y - \mu_y)^2}{n_y}$]


---

# Gaussian Naïve Bayes II

.col5050[
.col1[
**Learning model:**

| $y$  | $P(y)$ |
|:-----|-------:|
| Y    | 0.5    |
| N    | 0.5    |

<br>

|                 | Distance | Speed     |
|:----------------|---------:|----------:|
| $\mu_Y$         | 2.8      | 40.75     |
| $\mu_N$         | 39.25    | 43.95     |
| $\sigma_Y^2$    | 0.32     | 1734.605  |
| $\sigma_N^2$    | 2657.205 | 1653.125  |
]
.col2[
**Classification:**

$$P(Y|T)=0.5\cdot 0.0\cdot 0.00756 = 0.0$$

$$P(N|T)=0.5\cdot 0.00573\cdot 0.00722 = 0.00002$$

$$h(T)=N$$

**Note:**

$$P(speed=12.1|Y)=\frac{1}{\sqrt{2\cdot\pi\cdot 1734.605}}\cdot$$

$$\cdot \exp\left(-\frac{(12.1-40.75)^2}{2\cdot 1734.605}\right)=0.00669$$
]]

---

# Gaussian Naïve Bayes III

**Implementation:**

- Using LINQ of C#: 

  - [view](codes/gaussianNaiveBayes.html).red[*] / [output](codes/modelGNB.txt) / [download code](codes/gaussianNaiveBayes.cs)

- Using [sklearn](https://scikit-learn.org/stable/index.html) and [python](https://www.python.org/) in [colab](https://colab.research.google.com):

  - [view](codes/gnb.html) / [download](codes/gnb.ipynb)

- Using sklearn and Unity

  - sklearn: [view](codes/gnbjson.html) / [download](codes/gnbjson.ipynb) / [data](codes/tir.txt)

  - Unity: [view](codes/gnbjson2.html).red[*] / [download](codes/gnbjson.cs) / [model](codes/tir.json)

.footnote[.red[*] Formated with http://hilite.me/]


---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .cyan[Machine Learning]

  - .brown[Naïve Bayes]

  - .cyan[Decision Trees]

* Deep Learning

* Reinforcement Learning

* References

---

# Decision Trees

**Splitting measure:**

- Decision Trees: $accuracy=\frac{N(ok)}{N}$

- ID3: $entropy=-\sum_{\forall y} P(y) \cdot log_2(P(y))$

- CART: $gini = 1 - \sum_{\forall y} P(y)^2$ + binary trees

- .blue[Our approach]: $gini$ + k-ary trees

  - nominal and numeric features

  - classification and regression

  - one of the easiest decision trees

---

# Example I

.cols5050[
.col1[
| class      | cap-shape | cap-color |
|:-----------|:----------|:----------|
| poisonous  | convex    | brown     |
| edible     | convex    | yellow    |
| edible     | bell      | white     |
| poisonous  | convex    | white     |
| edible     | convex    | yellow    |
| edible     | bell      | white     |
| poisonous  | convex    | white     |
]
.col2[
**algorithm:**

each node of the tree from 

the minimum weighted sum of

 _Gini index_:

.small[$gini = 1 - \sum_{\forall y} P(y)^2$]
]]

---

# Example II

**cap-shape:**

| cap-shape | poisonous | edible | #examples |
|----------:|----------:|-------:|----------:|
| convex    | 3         | 2      | 5         |
| bell      | 0         | 2      | 2         |

.small[
.blue[For each _value_:]

$gini(\text{cap-shape}=\text{convex})=1-(\frac{3}{5})^2-(\frac{2}{5})^2=0.48$

$gini(\text{cap-shape}=\text{bell})=1-(\frac{0}{2})^2-(\frac{2}{2})^2=0.0$

.blue[_Weighted sum_:]

$gini(\text{cap-shape})=\frac{5}{7}\cdot 0.48+\frac{2}{7}\cdot 0.0=0.343$
]

---

# Example III

**cap-color:**

| cap-color | poisonous | edible | #examples |
|----------:|----------:|-------:|----------:|
| brown     | 1         | 0      | 1         |
| yellow    | 0         | 2      | 2         |
| white     | 2         | 2      | 4         |

.small[
.blue[For each _value_:]

$gini(\text{cap-color}=\text{brown})=1-(\frac{1}{1})^2-(\frac{0}{1})^2=0.0$

$gini(\text{cap-color}=\text{yellow})=1-(\frac{0}{2})^2-(\frac{2}{2})^2=0.0$

$gini(\text{cap-color}=\text{white})=1-(\frac{2}{4})^2-(\frac{2}{4})^2=0.5$

.blue[_Weighted sum_:]

$gini(\text{cap-color})=\frac{1}{7}\cdot 0.0+\frac{2}{7}\cdot 0.0+\frac{4}{7}\cdot 0.5=0.286$
]

---

# Example IV

**Selecting best feature:**

- best feature will be that with minimum _gini index_:

.small[
$$\text{best_feature}=\min((0.343,\text{cap-shape}),(0.286,\text{cap-color}))=\text{cap-color}$$
]

- every value with only a class will be a _leaf_:
  - brown $\rightarrow$ poisonous
  - yellow $\rightarrow$ edible

- a new set is built for the rest of values

.center[
| class      | cap-shape |
|:-----------|:----------|
| edible     | bell      |
| poisonous  | convex    |
| edible     | bell      |
| poisonous  | convex    |
_white_ examples without _cap-color_
]

- the process restarts with the new set

---

# Example V

**Resulting Tree:**

.center[
![:scale 55%](figures/dt.png)
]

### [chefboost](https://github.com/serengil/chefboost)

- [view](codes/dt1.html) / [download](codes/dt1.ipynb)

---

# Cutting Points I

**What about numerical attributes?**

| class      | width |
|:-----------|------:|
| versicolor | 2.9   |
| versicolor | 2.9   |
| virginica  | 3.0   |
| virginica  | 2.5   |

**Cutting points:**

| class      | width | cutting points | weighted ginis |
|:-----------|------:|---------------:|:---------------|
| virginica  | 2.5   |      |   | 
| versicolor | 2.9   | 2.7  | $\frac{1}{4}\cdot 0.0+\frac{3}{4}\cdot 0.45=0.0.3375$ |
| versicolor | 2.9   |      |   |
| virginica  | 3.0   | 2.95 | $\frac{3}{4}\cdot 0.45+\frac{1}{4}\cdot 0.0=0.0.3375$ |
.center[.small[cutting points for _width_ attribute]]

---

# Cutting Points II

**Gini example:**

| width | versicolor | virginica | #examples | gini |
|:------|-----------:|----------:|----------:|:-----|
| < 2.7 | 0 | 1 | 1 | $1-(\frac{0}{1})^2-(\frac{1}{1})^2=0$ |
| > 2.7 | 2 | 1 | 3 | $1-(\frac{2}{3})^2-(\frac{1}{3})^2=0.45$ |

.cols5050[
.col1[
**Resulting Tree:**

.center[
![:scale 80%](figures/dt-ct.png)
]
]
.col2[
### [chefboost](https://github.com/serengil/chefboost)

- [view](codes/dt2.html) / [download](codes/dt2.ipynb)

]]

---

# Regression

- standard deviation as splitting measure

$$\sigma=\sqrt{\frac{(x_1-\mu)^2+\dots+(x_n-\mu)^2}{n}}$$

.center[where $\mu=\frac{x_1+\dots+x_n}{n}$]

.cols5050[
.col1[
**example:**

| target | outlook | wind   |
|:-------|:--------|:-------|
| 25     | sun     | weak   |
| 30     | sun     | strong |
| 52     | rain    | weak   |
| 23     | rain    | strong |
| 45     | rain    | weak   |
.center[.small[[Source](https://sefiks.com/2018/08/28/a-step-by-step-regression-decision-tree-example/)]]
]
.col2[
**total amounts:**

$$\mu=35$$
$$\sigma=11.472$$
]]

---

# Regression II

.cols5050[
.col1[
**outlook:**

| outlook | $\mu$ | $\sigma$ | #examples |
|:--------|------:|---------:|----------:|
| sun     | 27.5  | 2.5      | 2         |
| rain    | 40.0  | 12.356   | 3         |
]
.col2[
**weighted sum:**

$$\sigma_{weighted}=\frac{2}{5}\cdot 2.5+\frac{3}{5}\cdot 12.356=8.414$$

**$\sigma$ reduction:**
$$\sigma_{reduction}=11.543-8.414=3.129$$
]]

.cols5050[
.col1[
**wind:**

| wind   | $\mu$  | $\sigma$ | #examples |
|:-------|-------:|---------:|----------:|
| weak   | 40.667 | 11.441   | 3         |
| strong | 26.5   | 3.5      | 2         |
]
.col2[
**weighted sum:**

$$\sigma_{weighted}=\frac{3}{5}\cdot 11.441+\frac{2}{5}\cdot 3.5=8.265$$

**$\sigma$ reduction:**
$$\sigma_{reduction}=11.543-8.265=3.278$$
]]

Wins the highest score: .blue[**wind**]

---

# Notes on Decision Trees

- Tends to overfitting when leafs with few examples

- High variance
  - small changes in training sets produce different trees

- **Prunning**: for avoiding overfitting
  - less than 5 instances 
  - maximum depth

- Previous regression tree by averaging leaf instances:

.cols5050[
.col1[ 
.center[![:scale 80%](figures/dtr.png)]
]
.col2[
### [chefboost](https://github.com/serengil/chefboost)

- [view](codes/dt3.html) / [download](codes/dt3.ipynb)

### sklearn (Projectile Motion)

- [view](codes/tir.html) / [download](codes/tir.ipynb)

]]

---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .brown[Machine Learning]

* .cyan[Deep Learning]

  - .cyan[Neural Networks]

  - Bias & Variance

  - DL Architectures

* Reinforcement Learning

* References

---

# Artificial Neuron Model

.center[![:scale 55%](figures/neuron.png)]

.center[![:scale 60%](figures/neuron_model.png)]

.footnote[Source: [Artificial Neuron](http://www.gabormelli.com/RKB/Artificial_Neuron)]

---

# Perceptron

.cols5050[
.col1[

- Classification and regression

- Linear model

- Classification:

$$h(x)=f(\sum_{i=1}^n w_i x_i + b)$$

- Learning rule:

$$w_i'=w_i+\eta(h(x)-y)$$
]
.col2[

![:scale 100%](figures/hyperplane.png)

]]

- Example in sklearn:
  - [view](codes/nn-sklearn.html) / [download](codes/nn-sklearn.ipynb) / [reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)

---

# Multi-layer Perceptron

.cols5050[
.col1[
- One hidden layer

- Non-linear model

- .blue[Classification & regression]

- .blue[Backpropagation]: <br>
Gradient descent: $W$ <br>
Loss function: $error(h(x),y)$

.center[![:scale 90%](figures/GradientDescent.png)]
.center[[source](http://tuxar.uk/brief-introduction-artificial-neural-networks/)]
]
.col2[
.center[![:scale 80%](figures/mlp.png)]
.center[[source](https://en.wikipedia.org/wiki/Artificial_neural_network)]
]]

---

# Examples

.blue[Splitting positives and negatives:]

.cols5050[
.col1[
![:scale 95%](figures/simpleNN.png)
]
.col2[
[Excel](codes/posNegNN.xlsx)

[Unity](codes/NNs.html) / [C#](codes/NNs.cs)
]]

.blue[sklearn on Iris:]

- [view](codes/nn-sklearn.html) / [download](codes/nn-sklearn.ipynb) 

---

# Example on Unity

.center[
![:scale 90%](figures/cars.gif)
[Building a neural network framework in C#](https://towardsdatascience.com/building-a-neural-network-framework-in-c-16ef56ce1fef)
]

[BackPropNetwork](https://github.com/kipgparker/BackPropNetwork)
- Unity Project in Github
- Backpropagation implementation


---

# Deep Learning

- Neural network with 2 or more hidden layers

.center[
![](figures/chart-1.png)

![](figures/chart-2.png)
]

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---

# Keras 

*Deep Learning high level library.*

.blue[Example] on MNIST:

  - [view](codes/keras-mlp.html) / [download](codes/keras-mlp.ipynb) / [Source](https://github.com/keras-team/keras)

Usual .blue[Parameters]:

| Task       | Loss function | Output Layer |
|:-----------|:--------------|:-------------|
| Bin. class | binary cross entropy | single unit, sigmoid activation |
| Multiclass | categorical cross entropy | one unit per class, softmax activation |
| Regression | MSE or RMSE | single unit, linear activation |

- Epochs: times all examples are trained
- Batch size: examples at each iteration
- Optimizer: gradient descent variants (AdaGrad, RMSProp, *Adam*)

[_Keras Documentation_](https://keras.io/)

---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .brown[Machine Learning]

* .cyan[Deep Learning]

  - .brown[Neural Networks]

  - .cyan[Bias & Variance]

  - DL Architectures

* Reinforcement Learning

* References

---

# Underfitting & Overfitting

.center[
![:scale 95%](figures/overfitting.png)
[source](https://tomrobertshaw.net/2015/12/introduction-to-machine-learning-with-naive-bayes/)
]

.cols5050[
.col1[
.blue[Underfitting](bias)

- Symptoms:
  - high training error
- Causes
  - model too simple
  - not enough training
]
.col2[
.blue[Overfitting](variance)
- Symptoms:
  - low training error
  - higher validation error
- Causes
  - model too complex
  - too much training
  - training set too small
]]

---

# Bias & Variance 

![](figures/bias.png)![](figures/early-stopping.png)

Normal use of a validation set to select parameters & avoid overfitting!

.footnote[Source: [left](https://towardsdatascience.com/regularization-the-path-to-bias-variance-trade-off-b7a7088b4577), [right](https://elitedatascience.com/overfitting-in-machine-learning)]

---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .brown[Machine Learning]

* .cyan[Deep Learning]

  - .brown[Neural Networks]

  - .brown[Bias & Variance]

  - .cyan[DL Architectures]

* Reinforcement Learning

* References

---

# Convolutional Neural Networks

**from Computer Vision**

to process image & video: .blue[invariant to translation & scale]

![:scale 90%](figures/cnn2.png)

.footnote[Source: [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)]

---

# Convolutional Neural Networks II

.cols5050[
.col1[
- Convolution: extract the high-level features such as edges
  - Learns the patterns

- Pooling: reduce dimensionality for 
  - max (edges) or average (photo)
  - computational cost
  - dominant features (rotational and positional invariant)

- Example:
  - Keras on MNIST
  - [view](codes/keras-cnn.html) / [download](codes/keras-cnn.ipynb) / [Source](https://github.com/keras-team/keras)
]
.col2[
![:scale 70%](figures/convolutional.gif)

![:scale 80%](figures/pooling.gif)
]]

.footnote[Source: [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)]

---

# The Neural Network Zoo

![:scale 90%](figures/zoo-1.png)

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---

# The Neural Network Zoo II

![:scale 90%](figures/zoo-2.png)

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---

# The Neural Network Zoo III

![:scale 90%](figures/zoo-3.png)

.footnote[Source: [The Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/)]

---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .brown[Machine Learning]

* .brown[Deep Learning]

* .cyan[Reinforcement Learning]

  - .cyan[Q-Learning]

  - RL Platforms

* References

---

# Reinforcement Learning

![:scale 65%](figures/rl.png)

.cols5050[
.col1[
- an _agent_
- a set of states $S$
- a set of actions $A$

]
.col2[
Learning a reward function $Q: S \times A \to \mathbb{R}$ for maximizing the total future reward.

]]

- _Q-Learning_: method for learning an aproximation of $Q$.

.footnote[Source: [My Journey Into Deep Q-Learning with Keras and Gym](https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762)]

---

# Q-table

Training example:

.center[![:scale 55%](figures/q-matrix.png)]

.footnote[Source: [Reinforcement Q-Learning from Scratch in Python with OpenAI Gym](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/)]

---

# Q-learning Example

- Simple example of Q-learning & Q-Table.

.center[[![:scale 65%](figures/ooops.png)](figures/ooops.mp4)]

- [Unity Package](codes/ooops.unitypackage) (in Spanish)
  - See the folder _Scripts_ in _Assets_

---

# Deep Reinforcement Learning

- Convolutional Neural Network for learning $Q$ <br>

.center[[![:scale 65%](figures/breakout.png)](https://www.youtube.com/watch?v=TmPfTpjtdgg&feature=youtu.be)]

.footnote[Source: [Deep Reinforcement Learning](https://deepmind.com/blog/article/deep-reinforcement-learning)]

---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .brown[Machine Learning]

* .brown[Deep Learning]

* .cyan[Reinforcement Learning]

  - .brown[Q-Learning]

  - .cyan[RL Platforms]

* References

---

# OpenAI Gym

**Example:** 

- CartPole from [OpenAi Gym](https://gym.openai.com/):

.center[![:scale 50%](figures/cartpole.gif)]

- [Deep Q-Learning with Keras and Gym](https://keon.io/deep-q-learning/)

---

# ML-Agents

Unity plugin for training intelligent agents.

[![:scale 70%](figures/mlagents.png)](https://www.youtube.com/watch?v=Hg3nmYD3DjQ&feature=youtu.be)

Address: https://github.com/Unity-Technologies/ml-agents

Contain .blue[Deep Learning] & .blue[Reinforcement Learning]


---
class: left, middle, inverse

# Outline

* .brown[Introduction]

* .brown[Machine Learning]

* .brown[Deep Learning]

* .brown[Reinforcement Learning]

* .cyan[References]

---

# References

- Gerard Escudero. [_Supervised Machine Learning_](https://gebakx.github.io/ml/). 2020. 

- Aurélien Géron. _Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow_, 2nd Edition. O'Reilly, 2019.

- Sefik Ilkin Serengil. [chefboost](https://github.com/serengil/chefboost) (2019): [_A Step by Step CART Decision Tree Example_](https://sefiks.com/2018/08/27/a-step-by-step-cart-decision-tree-example/) (2018). 

- Ian Millington. _AI for Games_ (3rd edition). CRC Press, 2019.

- [_Keras Documentation_](https://keras.io/). 

- DeepMind. [Deep Reinforcement Learning](https://deepmind.com/blog/article/deep-reinforcement-learning).

- Micheal Lanham. _Learn Unity ML-Agents - Fundamental of Unity Machine Learning_. Packt, 2018.

- Károly Zsolnai-Fehér. [OpenAI Plays Hide and Seek…and Breaks The Game!](https://www.youtube.com/watch?v=Lu56xVlZ40M). Two Minute Papers.

- Károly Zsolnai-Fehér. [DeepMind’s AlphaStar: A Grandmaster Level StarCraft 2 AI](https://www.youtube.com/watch?v=jtlrWblOyP4). Two Minute Papers.

- Siraj Raval. [How to Make an Amazing Video Game Bot Easily](https://www.youtube.com/watch?v=mGYU5t8MO7s&t=313s), 2016.


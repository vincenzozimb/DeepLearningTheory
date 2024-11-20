# Neural Networks and Gaussian Processes correspondence

## Introduction

```{admonition} Definition: Gaussian Processes (GP)
:class: tip
:name: def-GP
A function $f:\mathbb{R}^{N_0} \to \mathbb{R}$ is draw from a GP with mean function $\mu:\mathbb{R}^{N_0} \to \mathbb{R}$ and
kernel function $K:\mathbb{R}^{N_0} \times \mathbb{R}^{N_0} \to \mathbb{R}$ if, for every finite collection of inputs $\{x_1, \dots x_n\}$,
the vector of outputs $\{f(x_1), \dots f(x_n)\}$ is draw from a $n-$multivariate normal distribution with mean vector and covariance matrix
given with components:

$$
\mu_i = \mu(x_i), \qquad K_{ij} = K(x_i, x_j).
$$ 
```

Neal in the '90 {cite}`neal1996bayesian` discovered an equivalence between GPs and 1 HL (Hidden Layer) FC (Fully Connected) NNs (Neural Network) in the so called infinite-width limit (number of neurons in the layer that goes to infinity) and he also suggested that a similar correspondence might hold for DNNs (Deep NN), fact that was indeed proved to be true, see for example {cite}`google-brain`. In order to this correspondence to hold, it is necessary to choose particular prior over the parameters. 

```{admonition} Consequence
:class: important
Infinite-wide NN with gaussian iid prior over parameters $\iff$ GP prior over functions
```

## Calculation for 1 HL NN (Neal's result) 

Consider the multi-output (general) case. The NN output function $f(x)$ ($x \in \mathbb{R}^{N_0}, f(x)\in \mathbb{R}^{N_2}$) is defined as:

```{math}
    h_i(x) &= \sum_{j=1}^{N_0} w^{(1)}_{ij} x_j + b^{(1)}_i, & \quad \forall i=1,\dots,N_1, \\
    f_i(x) &= \sum_{j=1}^{N_1} w^{(2)}_{ij} \sigma(h_j(x)) + b^{(2)}_i, & \quad \forall i=1,\dots,N_2.
```

In the last expression, $\sigma:\mathbb{R} \to \mathbb{R}$ is a generic (usually non-linear) activation function. The infinite width limit is defined as $N_1 \to \infty$.

```{danger}
I did the calculations, but content is still to write!
```





# Bayesian Neural Networks

## Introduction

```{warning}
This content is temporary and will be edited/improved/corrected.
```


A NN is a function $x \mapsto f_\theta(x)$, where $\theta$ collectively denotes all the parameters.

Assume a probabilistic model where the output are rvs (random variables) and the data generating model is gaussian:

$$
y | x, \theta \sim \mathcal{N}(f_\theta(x), T)
$$

So the labels are not given directly by the output of the NN, but some noise is present. Physically speaking, we assume this noise to be regulated
by a parameter that we call "temperature", as usualy done in statistical mechanics.

The distribution over the inputs $x$ is not being modelled, so, from a bayesian point of view, $x$ can be formally be regarder as a hyperparameter.

Suppose to have a (training) dataset {math}`\mathcal{D}=\{ x_i, y_i \}_{i=1}^n`. In BNN (bayesian NN), the objective is to find the posterior predictive
distribution of $y_0 | x_0, \mathcal{D}$, where $x_0$ is a new (unseen) test data. This distribution is given by:

$$
p(y_0 | x_0, \mathcal{D}) &= \int d\theta ~ p(y_0, \theta| x_0, \mathcal{D}) \\
                         &= \int d\theta ~ p(y_0 | x_0, \theta, \cancel{\mathcal{D}}) p(\theta | \cancel{x_0}, \mathcal{D})
$$

- The first term does not depend on the dataset $\mathcal{D}$ because, as in any bayesian model, the observations (the $y_i$) are _conditionally independent_ given the parameters, while the inputs $x_i$ (that have to be regarded as hyperparameters) do not belong to the distribution of $y_0$ (only $x_0$ does).
- The second term, which is the **posterior distribution**, does not depend on $x_0$ because it is a hyperparameter for the data generating model of the yet _unobserved_ $y_0$.

The posterior distribution is given by Bayes' theorem:

$$
p(\theta | \mathcal{D}) \propto p(\theta) \prod_{i=1}^n p(y_i | x_i, \theta)
$$

Choosing the prior over the parameters $p(\theta)$ to be gaussian puts us in the situation where Neal's theory holds.

## Settings

The output function of a NN with $L$ HD is defined as:

$$
f_\theta(x) = \frac{1}{\sqrt{N_L}} \sum_{i_L=1}^{N_L} v_{i_L} \sigma(h_{i_L}^{(L)}(x)),
$$

where:

$$
h_{i_l}^{(l)}(x) &= \frac{1}{\sqrt{N_{l-1}}} \sum_{i_{l-1}=1}^{N_{l-1}} w_{i_l i_{l-1}}^{(l)} \sigma(h_{i_{l-1}}^{(l-1)}(x)), \qquad \forall l=2,\dots,L \\
h_{i_1}^{(1)}(x) &= \frac{1}{\sqrt{N_{0}}} \sum_{i_{0}=1}^{N_{0}} w_{i_1 i_0}^{(1)} x_{i_0}
$$

Without loss of generality, it is possible to neglect the biases, as it is always possible to include them in the weights by increasing by one the dimension of the input space. The symbol $\theta$ is a shorthand for all the parameters. 

## Using NNs for regression tasks

We are interested in (ridge) regression problems with a quadratic loss function:

$$
\mathcal{L} &= \frac{1}{2}\sum_{\mu=1}^{P} [y^\mu - f_\theta(x^\mu)]^2 + \mathcal{L}_{reg}, \\
\mathcal{L}_{reg} &= \frac{\lambda_L}{2\beta} ||v||^2 + \frac{1}{2\beta} \sum_{l=1}^{L-1} \lambda^{(l)} ||W^{(l)}||^2.
$$

In BNN we assume the Gibbs distribution for the parameters:

$$
p_\beta(\theta) = \frac{1}{\mathcal{Z}} e^{-\beta \mathcal{L}(\theta)} 
$$

And the normalization factor is called the partition function $\mathcal{Z}$, as usual in statistical mechanics:

$$
\mathcal{Z} = \int d\theta e^{-\beta \mathcal{L}(\theta)}
$$

```{note}
$$
\beta \mathcal{L} = \frac{\beta}{2}\sum_{\mu=1}^{P} [y^\mu - f_\theta(x^\mu)]^2 + \frac{\lambda_L}{2} ||v||^2 + \frac{1}{2} \sum_{l=1}^{L-1} \lambda^{(l)} ||W^{(l)}||^2
$$

implies that:

- $\beta \to \infty$ $(T \to 0)$ : enforce error minimization (i.e. stay at the bottom of the "energy" landscape)
- $\beta \to 0$ $(T \to \infty)$ : ignore the landscape, allow fluctuations around the minima. The only constraints are given by the regularization.

```

## Bayesian interpretation of the Gibbs probability:

$$
p_\beta(\theta) \propto \exp \left\{ -\frac{\beta}{2}\sum_{\mu=1}^{P} [y^\mu - f_\theta(x^\mu)]^2 \right\} 
    \exp \left\{ -\frac{\lambda_L}{2} ||v||^2 - \frac{1}{2} \sum_{l=1}^{L-1} \lambda^{(l)} ||W^{(l)}||^2 \right\}
$$

The second factor is proportional to the gaussian prior over the parameters, while the first represents the likelihood when we assume as a data 
generating model for the labels a gaussian distribution centered at the output value of the NN and with variance $T$, $y^\mu | x^\mu, \theta \sim \mathcal{N}(f_\theta(x^\mu), T)$.

```{note}
- $\beta \to \infty$ : training $\iff$ maximum likelihood estimation.
- $\beta < \infty$ : training $\iff$ maxumum a posteriori estimation.
```

In this framework, the average test error over a new (unseen) example is:

$$
\langle \varepsilon_g(x^0, y^0) \rangle = \int d\theta [y^0 - f_\theta(x^0)]^2 \frac{1}{\mathcal{Z}}e^{-\beta\mathcal{L}(\theta)}, 
$$

while the average training error on the dataset $\mathcal{D}$ at a given inverse temperature $\beta$:

$$
\langle \varepsilon_t \rangle = \int d\theta [\mathcal{L}(\theta) - \mathcal{L}_{reg}(\theta)] \frac{1}{\mathcal{Z}}e^{-\beta\mathcal{L}(\theta)}. 
$$
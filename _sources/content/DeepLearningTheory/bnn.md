# Bayesian Neural Networks

## Introduction

A NN is a function $x \mapsto f_\theta(x)$, where $\theta$ collectively denotes all the parameters.

Let {math}`\mathcal{D}=\{ x_i, y_i \}_{i=1}^n \equiv (\vec{x}, \vec{y})` be the (training) dataset.

Assume a probabilistic model where the output are rvs (random variables) and the data generating model is gaussian:

$$
y \mid x, \theta \sim \mathcal{N}(f_\theta(x), T)
$$

So the labels are not given directly by the output of the NN, but some noise is present. Physically speaking, we assume this noise to be regulated
by a parameter $T=\beta^{-1}$ that we call "temperature", as usualy done in statistical mechanics.

The distribution over the inputs $x$ is not being modelled, so, from a bayesian point of view, $x$ can be formally regarder as a hyperparameter.

We assume the parameters $\theta$ to be rvs, with some prior $p(\theta)$.

The posterior is then:

$$
p(\theta \mid \vec{y}, \vec{x}) \propto p(\vec{y} \mid \vec{x}, \theta) p(\theta \mid \cancel{\vec{x}}).
$$

The MAP (Maximum A Posteriori) estimator is:

\begin{equation*}
\hat{\theta}_{MAP} = \mathop{\mathrm{arg~max}}_{\theta} \left\{ -\frac{\beta}{2} ||\vec{y} - \vec{f}_\theta(\vec{x}) ||^2 + \log p(\theta) \right\}.
\end{equation*}

The calculation of $\hat{\theta}_{MAP}$ is equivalent to the minimization of the following loss function:

\begin{equation*}
\mathcal{L}(\theta;\mathcal{D}) = \frac{1}{2} ||\vec{y} - \vec{f}_\theta(\vec{x}) ||^2 - \frac{1}{\beta}\log p(\theta),
\end{equation*}

where the first term is the MSE (Mean Square Error) and the second acts as regularization, $\mathcal{L}_{reg}$.

Notice that the zero-temperature limit $\beta \to \infty$ corresponds to MLE (Maximum Likelihood Estimation), and to a regression task without regularization (i.e. enforce error minimization).

Using the loss function, the posterior distribution takes the form of a Gibbs distribution:

\begin{align*}
p(\theta \mid \mathcal{D}) &= \frac{1}{\mathcal{Z}} e^{-\beta \mathcal{L}(\theta;\mathcal{D})} & \mathcal{Z} &= \int d\theta e^{-\beta \mathcal{L}(\theta;\mathcal{D})}
\end{align*}

````{admonition} Information-theoretic interpretation of the Gibbs distribution
:class: dropdown tip

```{attention}
Content to write!
```

````

## Posterior-predictive

We are interested in finding the posterior predictive distribution of $y_0 \mid x_0, \mathcal{D}$, where $x_0$ is a new (unseen) test data.
This distribution is given by:

$$
p(y_0 \mid x_0, \mathcal{D}) &= \int d\theta ~ p(y_0, \theta\mid x_0, \mathcal{D}) \\
                         &= \int d\theta ~ p(y_0 \mid x_0, \theta, \cancel{\mathcal{D}}) p(\theta \mid \cancel{x_0}, \mathcal{D})
$$

- The first term does not depend on the dataset $\mathcal{D}$ because, as in any bayesian model, the observations (the $y_i$) are _conditionally independent_ given the parameters, while the inputs $x_i$ (that have to be regarded as hyperparameters) do not belong to the distribution of $y_0$ (only $x_0$ does).
- The second term, which is the **posterior distribution**, does not depend on $x_0$ because it is a hyperparameter for the data generating model of the yet _unobserved_ $y_0$.

The posterior distribution is given by Bayes' theorem:

$$
p(\theta \mid \mathcal{D}) \propto p(\theta) \prod_{i=1}^n p(y_i \mid x_i, \theta)
$$

This formulation is problematic because the posterior predictive is given as an integral in a large number of dimension!

## Posterior-predictive and Neal's theory

If we choose gaussian prior for the parameters, in the infinite-width limit Neal's theory (and its extension for deep NN) ensures that the output function of the NN is a gaussian process, $f \sim \mathcal{GP}(0, K)$, where $K$ is the NNGP kernel.

This means that, for every finite set of input values, like the $\vec{x}$ in the dataset $\mathcal{D}$, the output values $\vec{f} = [ f_\theta(x_i); i=1,\dots,n ]$ are normally distributed:

$$
\vec{f} \mid \vec{x} \sim \mathcal{N}(0, K_{\mathcal{D}\mathcal{D}}).
$$

$K_{\mathcal{D}\mathcal{D}}$ is the NNGP kernel calculated in the dataset.

If we consider a new (unseen) example, from the properties of GP follows that:

\begin{equation*}
\vec{f}, f_0 \mid \vec{x}, x_0 \sim \mathcal{N}\left( 0, 
    \begin{bmatrix} 
        K_{\mathcal{D}\mathcal{D}} & K_{\mathcal{D}x_0} \\
        K_{x_0\mathcal{D}} & K_{x_0x_0}    
    \end{bmatrix}
\right)
\end{equation*}

The posterior predictive for the network output is then:

\begin{align*}
p(f_0 \mid x_0, \mathcal{D}) &= \int d\vec{f} ~ p(f_0, \vec{f} \mid \vec{x}, \vec{y}) \\
&\propto \int d\vec{f} ~ p(\vec{y} \mid \vec{x}, \vec{f}, \cancel{f_0}) p(\vec{f}, f_0 \mid \vec{x}) & &\text{(Bayes theorem)} \\
&= \int d\vec{f} ~ \mathcal{N}_{\vec{y}}(\vec{f}(\vec{x}), T) ~ \mathcal{N}_{[\vec{f}, f_0]}\left( 0, 
    \begin{bmatrix} 
        K_{\mathcal{D}\mathcal{D}} & K_{\mathcal{D}x_0} \\
        K_{x_0\mathcal{D}} & K_{x_0x_0}    
    \end{bmatrix}
\right)
\end{align*}

This is a known integral in GP theory (see for example {cite}`Rasmussen2006Gaussian`), and the result is:

$$
f_0 \mid x_0, \mathcal{D} \sim \mathcal{N}(\bar{\mu}, \bar{K})
$$

with:

$$
\begin{cases}
\bar{\mu} = K_{x_0\mathcal{D}} \cdot \left( K_{\mathcal{D}\mathcal{D}} + \frac{1}{\beta}\mathbb{1} \right)^{-1} \cdot \vec{y} \\
\bar{K} = K_{x_0x_0} - K_{x_0\mathcal{D}} \cdot \left( K_{\mathcal{D}\mathcal{D}} + \frac{1}{\beta}\mathbb{1} \right)^{-1} \cdot K_{\mathcal{D}x_0}
\end{cases}
$$

````{admonition} Proof of the last result
:class: dropdown tip

```{warning}
Content still to write!
```

````

If the $y_0$ is needed, simply add to $f_0$ the noise, $y_0 = f_0 + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,T)$.

So, thanks to Neal's theory, it is much simpler to sample network outputs from the posterior predictive! We just need to calculate the NNGP kernel.


## Observables calculation

In the bayesian/stat-mech approach, the average test error over a new (unseen) example is:

$$
\langle \varepsilon_g(x_0, y_0) \rangle = \int d\theta [y_0 - f_\theta(x_0)]^2 \frac{1}{\mathcal{Z}}e^{-\beta\mathcal{L}(\theta)}, 
$$

while the average training error on the dataset $\mathcal{D}$ at a given inverse temperature $\beta$:

$$
\langle \varepsilon_t \rangle = \int d\theta [\mathcal{L}(\theta) - \mathcal{L}_{reg}(\theta)] \frac{1}{\mathcal{Z}}e^{-\beta\mathcal{L}(\theta)}. 
$$
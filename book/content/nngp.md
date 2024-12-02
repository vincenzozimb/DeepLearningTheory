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

**A mathematical detail**:

We also require for the finite collections of rvs to satisfy the natural *marginalization property*, 
that is, given any subset of a given (finite) collection of rvs, the covariance matrix calculated with 
the definition has to coincide with the relevant sub-matrix of the full covariance matrix. 
If this condition is satisfied, then the **Kolmogorov extention theorem** guarantees the existence of 
the stochastic process.  

```

Neal in the '90 {cite}`neal1996bayesian` discovered an equivalence between GPs and 1 HL (Hidden Layer) FC (Fully Connected) NNs (Neural Network) in the so called infinite-width limit (number of neurons in the layer that goes to infinity) and he also suggested that a similar correspondence might hold for DNNs (Deep NN), fact that was indeed proved to be true, see for example {cite}`google-brain`. In order to this correspondence to hold, it is necessary to choose particular prior over the parameters. 

```{admonition} Consequence
:class: note
Infinite-wide NN with gaussian iid prior over parameters $\iff$ GP prior over functions
```

## Calculation for 1 HL NN (Neal's result) 

Consider the multi-output (general) case. The NN output function $f(x)$ ($x \in \mathbb{R}^{N_0}, f(x)\in \mathbb{R}^{N_2}$) is defined as:

```{math}
    h_i(x) &= \sum_{j=1}^{N_0} w^{(1)}_{ij} x_j + b^{(1)}_i, & \quad \forall i=1,\dots,N_1, \\
    f_i(x) &= \sum_{j=1}^{N_1} w^{(2)}_{ij} \sigma(h_j(x)) + b^{(2)}_i, & \quad \forall i=1,\dots,N_2.
```

In the last expression, $\sigma:\mathbb{R} \to \mathbb{R}$ is a generic (usually non-linear) activation function. The infinite width limit is defined as $N_1 \to \infty$.

Choose IID gaussian priors over the parameters, as follows:

\begin{align*}
w_{ij}^{(l)} &\sim \mathcal{N}(0, \sigma_w^2/N_{l-1}), & b_i^{(l)} &\sim \mathcal{N}(0, \sigma_b^2), & l&=1,2.
\end{align*}

```{important}
This choice of NN parametrization is referred to as the *standard parametrization*. An alternative is the *NTK parametrization*, which consists in extracting a factor $\sigma_w/\sqrt{N_l}$ in front of the layer-tp-layer transformation and to sample the parameters from a $\mathcal{N}(0,1)$. 

Both schemes give rise to the same Gaussian processes in the infinite-width limit, but in general there are differences in their dynamics under gradient descent. See the literature for more details.
```



As parameters are rvs and data (inputs) are fixed, it follows that the pre-activations $h$ and the output values $f$ are also rvs. 

The pre-activations are independent and normally distributed:

```{admonition} Formula
:class: dropdown tip

Sum of gaussian rvs is a gaussian rv:

$$
\sum_{i=1}^{n}\mathcal{N}(\mu_i, \sigma_i^2) = \mathcal{N}\left(\sum_{i=1}^{n} \mu_i, \sum_{i=1}^{n} \sigma_i^2\right) 
$$

```

\begin{align*}
h_1,\dots,h_{N_1} &\mathop{\sim}^{IID} \mathcal{N}\left( 0, \frac{\sigma_w^2}{N_0} \sum_j^{N_0} x_j^2 + \sigma_b^2 \right)
\end{align*}

In particular, the pre-activations are independent rvs. 

It follows that also the post-activations $\sigma_i \equiv \sigma(h_i)$ are IID rvs.

```{admonition} Mathematical fact
:class: dropdown tip

Functions of independent rvs are independent rvs.

\begin{align*}
    X_i &\sim p_{X_i} & Y_i &:= f_i(X_i) 
\end{align*}

*proof:*

\begin{align*}
    p_{Y_1Y_2}(y_1,y_2) &= \int dx_1dx_2 ~ \underbrace{p_{X_1X_2}(x_1,x_2)}_{p_{X_1}(x_1)p_{X_2}(x_2)} \delta(y_1-f_1(x_1)) \delta(y_2-f_2(x_2)) \\
    &= \left[\int dx_1 ~ p_{X_1}(x_1) \delta(y_1-f_1(x_1)) \right] \left[ \int dx_2 ~ p_{X_2}(x_2) \delta(y_2-f_2(x_2)) \right] \\
    &= p_{Y_1}(y_1) p_{Y_2}(y_2). & \square
\end{align*}

```

The same is true also for all the product terms $w^{(2)}\sigma$, that are themselves IID, as product of IID terms (although their probability distribution can be rather complex). 

In the limit $N_1 \to \infty$, the CLT (Central Limit Theorem) ensures that the sum $\sum_{j}^{N_1} w_{ij}^{(2)}\sigma_j$ is asymptotically normal. The same holds also when the contribution from the biases $b^{(2)}$ is taken into account (as again, sum of normal rvs is a normal rv). Likewise, for any finite set of inputs, the sets of the outputs (for each component) of a NN in the infinite width limit will also be normally distributed (multivariate CLT). This is exactly the definition of Gaussian Process.

So it was proven that $f_i \sim \mathcal{GP}(\mu, K) ~ \forall i=1,\dots,N_2$. The parameters of the Gaussian process are all the same, and are calculated as follows.

- The mean function is:

$$
\mu(x) = \mathbb{E}[f_i(x)] = 0 \qquad \forall i.
$$

As each component $f_i$ has zero mean, it follows from a simple calculation exploiting independence of the involved rvs.

- The kernel function, called the NNGP (Neural Network Gaussian Process) kernel is (again, $\forall i$):

\begin{align*}
K(x, x') &= \mathbb{E}[f_i(x), f_j(x)] - \cancel{\mathbb{E}[f_i(x)]}\cancel{\mathbb{E}[f_i(x')]} \\
&= \mathbb{E}\left[ \sum_{jj'}^{N_1} w_{ij}^{(2)} w_{ij'}^{(2)} \sigma_j \sigma_{j'} + b_i^{(2)} \sum_{j'}^{N_1} w_{ij'}^{(2)}\sigma_{j'} + b_{i}^{(2)} \sum_{j}^{N_1} w_{ij}^{(2)}\sigma_{j} + b_i^{(2)2} \right] \\
&= \sum_{jj'}^{N_1} \delta_{jj'} \frac{\sigma_w^2}{N_1} C(x,x') + \sigma_b^2 \\
&= \sigma_w^2 C(x,x') + \sigma_b^2.
\end{align*}

The last passage is valid as the newly defined quantity:

$$
C(x,x') := \mathbb{E}[\sigma(h_j(x)) \sigma(h_j(x'))]
$$

does not depend on $j$ (remember that the $h$ are IID).

This conclude the proof, as the statement $f\sim\mathcal{GP}$ was proven.

---

Note that:

\begin{align*}
K^0(x, x') &:= \mathbb{E}[h_i(x), h_j(x)] - \cancel{\mathbb{E}[h_i(x)]}\cancel{\mathbb{E}[h_i(x')]} \\
&= \sum_{jj'}^{N_0} \delta_{jj'} \frac{\sigma_w^2}{N_0} x_jx_{j'}' + \sigma_b^2 \\
&= \sum_{j}^{N_0} \frac{\sigma_w^2}{N_0} x_jx_j' + \sigma_b^2 \\
&= \sigma_w^2 \left( \frac{x \cdot x'}{N_0} \right) + \sigma_b^2.
\end{align*}

This is nothing but the covariance matrix of the $h$ variables, 
when calculated at two different dataset points, $x$ and $x'$. We can use this result to reduce the calculation of each element $C(x,x')$ to a two-dimensonal integral, as follows:

\begin{align*}
C(x,x') &= \mathbb{E}[\sigma(u)\sigma(u')] & &\text{with } u,u'\sim \mathcal{N}(0,\Sigma)
\end{align*}

Where:

$$
\Sigma = 
\begin{pmatrix}
K^0(x,x) & K^0(x,x') \\
K^0(x',x) & K^0(x',x')
\end{pmatrix}
$$


## Extension to Deep Neural Networks

The proof procedes by induction. The final kernel is calculated recursively using the kernel at the previous HL. The result is as expected (for a proof see for example {cite}`google-brain`):

$$
\begin{cases}
K^0(x,x') &= \sigma_b^2 + \sigma_w^2 \frac{x \cdot x'}{N_0} \\
K^l(x,x') &= \sigma_b^2 + \sigma_w^2 \mathbb{E}[\sigma(u)\sigma(u')] & &\text{with } u,u'\sim \mathcal{N}(0,\Sigma^l)
\end{cases}
$$

where:

$$
\Sigma^l = 
\begin{pmatrix}
K^{l-1}(x,x) & K^{l-1}(x,x') \\
K^{l-1}(x',x) & K^{l-1}(x',x')
\end{pmatrix}
$$
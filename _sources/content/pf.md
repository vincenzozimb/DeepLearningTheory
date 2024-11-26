# Partition function of 1 HL FC NN (single output)


## Introduction

Paper {cite}`Pacelli_2023`

## Notations:

- $i_k$ runs from $1$ to $N_k$ ($k=0,1$). $N_0$ is the dimension of the input space while $N_1$ the width of the hidden layer. 
- Greek indexes $\mu, \nu$ from $1$ to $P$, where $P$ is the size of the dataset.
- Einstein summation convention is assumed on the $i_k$ unless otherwise specified, but not on $\mu,\nu$.
- $Dv = \prod_{i_1} dv_{i_1}, \quad Dw = \prod_{i_0i_1} dw_{i_1i_0}, \quad Dh^\mu = \prod_{i_1} dh_{i_1}^\mu$. 

The output of the NN is given by:

$$
f_\theta(x^\mu) = \frac{1}{\sqrt{N_1}} v_{i_1} \sigma \left( \frac{1}{\sqrt{N_0}} w_{i_1i_0}x_{i_0}^\mu \right),
$$

where $\sigma$ is the activation function.

The loss function is given by:

$$
\beta \mathcal{L} = \frac{\beta}{2}\sum_\mu (y^\mu - f_\theta(x^\mu))^2 + \frac{\lambda_1}{2}v_{i_1}v_{i_1} +\frac{\lambda_0}{2}w_{i_1i_0}w_{i_1i_0}.
$$

## Full calculation:

As a first step, to decouple the nested integrals over the weigths, let us introduce delta functions over the preactivations and the output values, as follows:

$$
\mathcal{Z} &= \int DvDw \exp \left\{ -\frac{\beta}{2}\sum_\mu (y^\mu - f_\theta(x^\mu))^2 - \frac{\lambda_1}{2}v_{i_1}v_{i_1} -\frac{\lambda_0}{2}w_{i_1i_0}w_{i_1i_0} \right\} \\
% ------------------------------------------------------------------------------------
&= \int DvDw \left(\prod_\mu ds^\mu Dh^\mu\right)  \exp \left\{ -\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 - \frac{\lambda_1}{2}v_{i_1}v_{i_1} -\frac{\lambda_0}{2}w_{i_1i_0}w_{i_1i_0} \right\} \times \\
& \qquad \times \prod_\mu \delta \left( s^\mu - \frac{v_{i_1}\sigma(h_{i_1}^\mu)}{\sqrt{N_1}} \right) \prod_{\mu,i_1} \delta \left( h_{i_1}^\mu - \frac{w_{i_1i_0}x_{i_0}^\mu}{\sqrt{N_0}} \right).
$$

Now insert the integral representation of the Dirac deltas:

$$
&= \int DvDw \left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \frac{Dh^\mu D\bar{h}^\mu}{(2\pi)^{N_1}}\right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 - \frac{\lambda_1}{2}v_{i_1}v_{i_1} -\frac{\lambda_0}{2}w_{i_1i_0}w_{i_1i_0}\right] \times \\
& \qquad \times \exp\left(i\sum_\mu s^\mu \bar{s}^\mu -i \frac{v_{i_1}}{\sqrt{N_1}}\sum_\mu \bar{s}^\mu \sigma(h_{i_1}^\mu)\right) \exp\left(i\sum_\mu h_{i_1}^\mu \bar{h}_{i_1}^\mu -i \frac{w_{i_1i_0}}{\sqrt{N_0}}\sum_\mu \bar{h}_{i_1}^\mu x_{i_0}^\mu\right).
$$

### Integrals over the weigths

Now the integrals over the weigths can be isolated. Let us calculate them separately.

- Integral over the $v$ variables:

$$
&\int Dv ~ \exp \left[-\frac{\lambda_1}{2}v_{i_1}v_{i_1}\right] \exp \left[-i\frac{v_{i_1}}{\sqrt{N_1}}\sum_\mu \bar{s}^\mu \sigma(h_{i_1}^\mu) \right] = \\
=& \prod_{i_1} \int dv_{i_1} ~ \exp \left[-\frac{\lambda_1}{2}v_{i_1}^2\right] \exp \left[-i\frac{v_{i_1}}{\sqrt{N_1}}\sum_\mu \bar{s}^\mu \sigma(h_{i_1}^\mu) \right] & \qquad \text{(no sum)} \\
=& \left( \frac{2\pi}{\lambda_1} \right)^{N_1/2} \exp \left[ -\frac{1}{2\lambda_1 N_1}\sum_{\mu\nu}\bar{s}^\mu \bar{s}^\nu \sigma(h_{i_1}^\mu)\sigma(h_{i_1}^\nu) \right]
$$

- Integral over the $w$ variables:

$$
&\int Dw ~ \exp \left[-\frac{\lambda_0}{2}w_{i_1i_0}w_{i_1i_0}\right] \exp \left[-i\frac{w_{i_1i_0}}{\sqrt{N_0}}\sum_\mu \bar{h}_{i_1}^\mu x_{i_0}^\mu \right] = \\
=& \prod_{i_0i_1} \int dw_{i_1i_0} ~ \exp \left[-\frac{\lambda_0}{2}w_{i_1i_0}^2\right] \exp \left[-i\frac{w_{i_1i_0}}{\sqrt{N_0}}\sum_\mu \bar{h}_{i_1}^\mu x_{i_0}^\mu \right] & \qquad \text{(no sum)} \\
=& \left( \frac{2\pi}{\lambda_0} \right)^{N_0N_1/2} \exp \left[ -\frac{1}{2\lambda_0 N_0}\sum_{\mu\nu}\bar{h}_{i_1}^\mu \bar{h}_{i_1}^\nu x_{i_0}^\mu x_{i_0}^\nu \right]
$$

Define $A:=(2\pi/\lambda_1)^{N_1/2} (2\pi/\lambda_0)^{N_0N_1/2}$. The partition function is then:

$$
\mathcal{Z} &= A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \frac{Dh^\mu D\bar{h}^\mu}{(2\pi)^{N_1}}\right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu (s^\mu \bar{s}^\mu + h_{i_1}^\mu \bar{h}_{i_1}^\mu)  \right] \times \\
& \qquad \times \exp \left[ -\frac{1}{2\lambda_1 N_1}\sum_{\mu\nu}\bar{s}^\mu \bar{s}^\nu \sigma(h_{i_1}^\mu)\sigma(h_{i_1}^\nu) -\frac{1}{2\lambda_0 N_0}\sum_{\mu\nu}\bar{h}_{i_1}^\mu \bar{h}_{i_1}^\nu x_{i_0}^\mu x_{i_0}^\nu \right]
$$

Notice that the integrals over the $h$ variables are all equals, so:

$$
\mathcal{Z} &= A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] \times \\
& \qquad \times \left[ \int \left(\prod_\mu \frac{dh^\mu d\bar{h}^\mu}{2\pi}\right) \exp \left( i\sum_\mu h^\mu \bar{h}^\mu -\frac{1}{2\lambda_1 N_1}\sum_{\mu\nu}\bar{s}^\mu \bar{s}^\nu \sigma(h_{i_1}^\mu)\sigma(h_{i_1}^\nu) -\frac{1}{2\lambda_0 N_0}\sum_{\mu\nu}\bar{h}_{i_1}^\mu \bar{h}_{i_1}^\nu x_{i_0}^\mu x_{i_0}^\nu \right) \right]^{N_1}
$$


### Gram matrix

Let us introduce the $P \times P$ matrix with elements (remember the sum over $i_0$):

$$
C_{\mu\nu} := \frac{1}{\lambda_0N_0} x_{i_0}^\mu x_{i_0}^\nu \propto \boldsymbol{x}^\mu \cdot \boldsymbol{x}^\nu =: (X^t X)_{\mu\nu}.
$$

In the last expression I introduced the so called Gram matrix $X^tX$ , defined in terms of the $N_0 \times P$ matrix $X=(\boldsymbol{x}^1, \dots \boldsymbol{x}^P)$. 

- If $P>N_0$: 

$$
rank(X^tX) \le \min \{rank(X^t),rank(X) \} = rank(X) \le N_0 < P
$$

So $C$ cannot have maximum rank and therefore it is not invertible (intuitively, there are more vectors than dimensons, they cannot be linearly independent).

- If $P \le N_0$:

$C$ is invertible if and only if the data are linearly independent.

In any case, when $C$ is not invertible, it is always possible to calculate $(C+\varepsilon \mathbb{1})^{-1}$ for some small $\varepsilon>0$ and check that the final results are independent from $\varepsilon$ (as long as we choose it to be small enough).


### Going back to the partition function

Let us calculate the integral in the $\bar{h}$ variables:

$$
&\int \left( \prod_\mu \frac{d\bar{h}^\mu}{2\pi} \right) \exp \left(i\sum_\mu h^\mu \bar{h}^\mu \right) \exp \left(-\frac{1}{2}\sum_{\mu\nu}\bar{h}^\mu C_{\mu\nu}\bar{h}^\nu \right) \\
&=[(2\pi)^P \det C]^{-1/2} \exp \left(-\frac{1}{2}\sum_{\mu\nu} h^\mu C_{\mu\nu}^{-1} h^\nu \right) \\
&=: P(\{ h^\mu \}) \qquad \qquad \text{(a normalized gaussian)}
$$

So the partition function is:

$$
\mathcal{Z} &= A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] \times \\
& \qquad \times \left[ \int \left(\prod_\mu dh^\mu \right) \exp \left(-\frac{1}{2\lambda_1 N_1}\sum_{\mu\nu}\bar{s}^\mu \bar{s}^\nu \sigma(h_{i_1}^\mu)\sigma(h_{i_1}^\nu) \right) P(\{ h^\mu \}) \right]^{N_1}
$$

Let us introduce a new variable $q$:

$$
\mathcal{Z} &= A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] \times \\
& \qquad \times \left[ \int \left(\prod_\mu dh^\mu \right) dq ~ e^{-q^2/2} P(\{ h^\mu \}) \delta \left( q - \frac{1}{\sqrt{\lambda_1N_1}}\sum_\mu \bar{s}^\mu \sigma(h^\mu) \right) \right]^{N_1}
$$

The integral over the $h^\mu$ variables gives, by definition, the probability density $P(q)$.


```{admonition} Formula
:class: dropdown tip
The following property is a consequence of the change of variables theorem for random variables:

$$
X \sim p_X \qquad Y=f(X) \qquad \Longrightarrow \qquad Y \sim p_Y
$$

Where:

$$
    p_Y(y) = \int dx ~ p_X(x) \delta(y-f(x)).
$$

```

$$
\mathcal{Z} = A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] \times \left[ \int dq ~ e^{-q^2/2} P(q) \right]^{N_1}
$$


### Distribution of $q$

Recall that the variable $q$ is defined as:

$$
q = \frac{1}{\sqrt{\lambda_1N_1}}\sum_\mu \bar{s}^\mu \sigma(h^\mu).
$$

Its mean value is:

$$
\mathbb{E}[q] = \frac{1}{\sqrt{\lambda_1N_1}}\sum_\mu \bar{s}^\mu \mathbb{E}[\sigma(h^\mu)].
$$

```{attention}
The last passage is valid only because the $\bar{s}$ variable has to be integrated after, this expression is inside the integrals in $d\bar{s}$.
Otherwise, the $s$ variables themselfes would depend on the $h$ variables, and the calculation would not be valid.
```

#### Zero-mean activation functions

We assume to work with zero-mean activation functions, that are activation functions whose mean according to a centered gaussian distribution is zero.
Note that ReLU is not in this class.

\begin{align*}
\mathbb{E}[\sigma(h^\mu)] &= \int \left( \prod_\mu dh^\mu \right) P(\{ h^\mu \}) \sigma(h^\mu) = 0 & \Longrightarrow & &\mathbb{E}[q]=0.
\end{align*}

---

Now calculate the variance:

$$
\mathbb{E}[q^2] &= \frac{1}{\lambda_1N_1} \sum_{\mu\nu}\bar{s}^\mu \bar{s}^\nu \mathbb{E}[\sigma(h^\mu)\sigma(h^\nu)] \\
&\equiv \frac{1}{\lambda_1N_1} \sum_{\mu\nu}\bar{s}^\mu K_{\mu\nu} \bar{s}^\nu =: Q(\bar{s}, C),
$$

where $K$ is the NNGP kernel! (see the page {doc}`nngp`).


#### Breuer-Major theorem

Assuming that the BM theorem holds:

\begin{align*}
q &\to \mathcal{N}(0, Q(\bar{s}, C)) & &\text{for } P,N_1 \to \infty, P/N_1=\alpha_1.
\end{align*}

Note that the proportional limit is the appropriate limit to invoke the BM theorem, as the sum is among $P$ terms but the prefactor is $1/\sqrt{N_1}$.

So, in this limit, we can perform the integral in $q$ and the result is:

$$
\int dq ~ P(q) e^{-q^2/2} = (1+Q(\bar{s}, C))^{-1/2}.
$$

In particular we need $Q \ge -1$ (always satisfied as it is a variance).


### Continuing the calculation

$$
\mathcal{Z} = A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] (1+Q(\bar{s}, C))^{-N_1/2}.
$$

Introduce a Dirac delta over the function $Q(\bar{s}, C)$ and also its integral representation:

\begin{align*}
\mathcal{Z} &= A \int \frac{dQd\bar{Q}}{2\pi} e^{iQ\bar{Q}} (1+Q)^{-N_1/2} \times \\
&\qquad \times \int \left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu -i\bar{Q} \frac{1}{\lambda_1N_1} \sum_{\mu\nu}\bar{s}^\mu K_{\mu\nu} \bar{s}^\nu \right].
\end{align*}

Let's focus on the integral over the $s$ variables, which is the inverse Fourier transform of a gaussian:

\begin{align*}
\int &\left(\prod_\mu ds^\mu \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] = \\
=& e^{i\sum_\mu \bar{s}^\mu y^\mu} \left( \frac{2\pi}{\beta} \right)^{P/2} \exp \left[ -\frac{1}{2\beta} \sum_\mu \bar{s}^{\mu^2} \right].
\end{align*}

Now the integral over the $\bar{s}$ becomes:

\begin{align*}
\int &\left(\prod_\mu \frac{d\bar{s}^\mu}{2\pi} \right) \exp \left[i\sum_\mu \bar{s}^\mu y^\mu -\frac{1}{2\beta} \sum_\mu \bar{s}^{\mu^2} -i\bar{Q} \frac{1}{\lambda_1N_1} \sum_{\mu\nu}\bar{s}^\mu K_{\mu\nu} \bar{s}^\nu \right] \\
=& \int \left(\prod_\mu \frac{d\bar{s}^\mu}{2\pi} \right) \exp \left[i\sum_\mu \bar{s}^\mu y^\mu -\frac{1}{2} \sum_{\mu\nu} \bar{s}^{\mu} \left( \frac{2i\bar{Q}}{\lambda_1N_1}K_{\mu\nu} + \frac{1}{\beta}\delta_{\mu\nu} \right) \bar{s}^\nu \right] \\
=& \left[ (2\pi)^P \det \left( \frac{2i\bar{Q}}{\lambda_1N_1}K + \frac{1}{\beta}\mathbb{1} \right) \right]^{-1/2} \exp \left[ -\frac{1}{2}\sum_{\mu\nu} y^\mu \left( \frac{2i\bar{Q}}{\lambda_1N_1}K + \frac{1}{\beta}\mathbb{1} \right)^{-1} y^\nu \right].
\end{align*}

The last passage was done recognizing that the integral is again the inverse Fourier transform of a gaussian.

So we have:

\begin{align*}
\mathcal{Z} &= A  \left( \frac{\cancel{2\pi}}{\beta} \right)^{P/2} (\cancel{2\pi})^{-P/2} \int \frac{dQd\bar{Q}}{2\pi} e^{iQ\bar{Q}} (1+Q)^{-N_1/2} \times \\
&\qquad \times \left[ \det \left( \frac{2i\bar{Q}}{\lambda_1N_1}K + \frac{1}{\beta}\mathbb{1} \right) \right]^{-1/2} \exp \left[ -\frac{1}{2}\sum_{\mu\nu} y^\mu \left( \frac{2i\bar{Q}}{\lambda_1N_1}K + \frac{1}{\beta}\mathbb{1} \right)^{-1} y^\nu \right].
\end{align*}

Now change variable $\bar{Q} \to -iN_1\bar{Q}/2$. This makes the integral complex, and $\bar{Q}$ now runs from $-i\infty$ to $i\infty$. However, as the integrand is holomorphic, the same result is obtained deforming the integration path back to a real one.

\begin{align*}
\mathcal{Z} &= \frac{A}{2\pi} (\beta)^{-P/2} \frac{N_1}{2} \int dQd\bar{Q} e^{\frac{N_1}{2}Q\bar{Q}} (1+Q)^{-N_1/2} ( \det \tilde{K} )^{-1/2} \exp \left( -\frac{1}{2}\sum_{\mu\nu} y^\mu \tilde{K}^{-1} y^\nu \right).
\end{align*}

Where we have defined:

$$
\tilde{K} := \frac{\bar{Q}}{\lambda_1}K + \frac{1}{\beta}\mathbb{1}
$$

Rearranging (remember that $\alpha_1=P/N_1$):

```{admonition} Formula
:class: dropdown tip

Determinant of a matrix:

$$
\log \det M = \text{Tr} \log M,
$$

and:

$$
    \text{Tr} \log (AB) = \text{Tr} \log A + \text{Tr} \log B.
$$

```

\begin{align*}
\mathcal{Z} &= \frac{A}{2\pi} \frac{N_1}{2} \int dQd\bar{Q} \exp \left( -\frac{N_1}{2} S[Q,\bar{Q}] \right)
\end{align*}

where the action $S[Q,\bar{Q}]$ is given by:

$$
S[Q,\bar{Q}] = -Q\bar{Q} + \log(1+Q) + \frac{\alpha_1}{P} \text{Tr} \log(\beta\tilde{K}) + \frac{\alpha_1}{P} \boldsymbol{y}^t \cdot \tilde{K}^{-1} \cdot \boldsymbol{y} 
$$

```{warning}
- The factor $A$ contains the normalization factors for the prior distribution. However, it should not be present (it can be eliminated using the proper integration measure for the weigths, i.e. including them in the partition function definition?).

- Neither the factor $N_1/4\pi$ should be present (potentially, I do not care about additional constants in $\log \mathcal{Z}$, the only issue is that $N_1 \to \infty$. So $\mathcal{Z}$ would be the "regularized" partition function).
```

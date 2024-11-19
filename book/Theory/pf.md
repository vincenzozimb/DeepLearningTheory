# Partition function of 1 HL FC NN

## Introduction

Paper {cite}`Pacelli_2023`


## Partition function calculation for a 1 HL FC NN, single output

### Notations:

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

### Partition function calculation:

As a first step, to decouple the nested integrals over the weigths, let us introduce delta functions over the preactivations and the output values, as follows:

````{div} full-width

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
& \qquad \times \left[ \int \left(\prod_\mu \frac{Dh^\mu D\bar{h}^\mu}{2\pi}\right) \exp \left( i\sum_\mu h^\mu \bar{h}^\mu -\frac{1}{2\lambda_1 N_1}\sum_{\mu\nu}\bar{s}^\mu \bar{s}^\nu \sigma(h_{i_1}^\mu)\sigma(h_{i_1}^\nu) -\frac{1}{2\lambda_0 N_0}\sum_{\mu\nu}\bar{h}_{i_1}^\mu \bar{h}_{i_1}^\nu x_{i_0}^\mu x_{i_0}^\nu \right) \right]^{N_1}
$$

#### Gram matrix

Let us introduce the $P \times P$ matrix with elements (remember the sum over $i_0$):

$$
C_{\mu\nu} := \frac{1}{\lambda_0N_0} x_{i_0}^\mu x_{i_0}^\nu.
$$



````
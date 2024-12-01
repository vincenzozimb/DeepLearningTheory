# Gradient Descent dynamics and Neural Tangent Kernel 

## Notation

- Adopt Einstein summation convention on repeated indeces.
    - $i,j$ run over the components of the parameters $\theta$.
    - $a,b$ run over the components of the output function $f$.
    - $\mu,\nu$ run over the dimensions of the dataset $\mathcal{D}$.


## Introduction

A NN is a function $f(\cdot;\theta):\mathbb{R}^{n_0} \to \mathbb{R}^k$ (multiple outputs if $k>0$), defined in vector notation as (for $L$ hidden layers):

$$
f(x,\theta) = W^{(L)} \cdot \sigma(h^{(L)}(x)) + b^{(L)},
$$

where:

$$
\begin{cases}
h^{(1)} = W^{(1)} \cdot x + b^{(1)} \\
h^{(l)} = W^{(l)} \cdot \sigma(h^{(l-1)(x)}) + b^{(l)} & \qquad 1<l\le L
\end{cases}
$$

Let $\mathcal{D} = \{ (x^\mu, y^\mu) \}_{\mu=1}^P \equiv \mathcal{D}_x \cup \mathcal{D}_y$ be the (training) dataset, and consider a MSE loss function:

$$
\mathcal{L}(\theta) = \frac{1}{2} \sum_{\mu=1}^P || f(x^\mu;\theta) - y^\mu ||^2.
$$

As a function of $\theta$, $\mathcal{L}$ is a very complicated function, but it's rather simple as a function of the output values on the dataset $z^\mu(\theta):=f(x^\mu;\theta)$.

Define the vector (of vectors if $k>1$)

$$
z(\theta) := (z^\mu(\theta), \mu=1,\dots,P)
$$

Under a (continuous time) gradient descent dynamics $\dot{\theta}(t) = -\eta \nabla_\theta \mathcal{L}$:

$$
\dot{\theta}_a(t) = -\eta \frac{\partial \mathcal{L}}{\partial \theta_a} \bigg\vert_{\theta(t)} = -\eta \frac{\partial \mathcal{L}}{\partial z_i} \bigg\vert_{z(\theta(t))} \frac{\partial z_i}{\partial \theta_a} \bigg\vert_{\theta(t)}
$$

the output values evolve according to:

$$
\dot{z}_i(\theta(t)) = -\eta \frac{\partial \mathcal{L}}{\partial z_j} \bigg\vert_{z(\theta(t))} \underbrace{\frac{\partial z_j}{\partial \theta_a} \bigg\vert_{\theta(t)} \frac{\partial z_i}{\partial \theta_a} \bigg\vert_{\theta(t)}}_{\Theta_{ji}(t)} + \mathcal{o}(\eta^2) 
$$

The newly defined quantity $\Theta$ is called the NTK (Neural Tangent Kernel). Under gradient descent dynamics, it controls the evolution of the output values of a NN according to:
 
$$
\dot{z}(t) = -\eta \Theta(t) \cdot \nabla_z \mathcal{L} \bigg\vert_{z(\theta(t))} + \mathcal{o}(\eta^2)
$$

Suppose to have the NTK, we could then solve the last equation (it's a system of ODE, even linear in the case of a MSE loss function) and obtain directly the evolution of the NN output function during GD!

The issue is that calculating the NTK is not an easy task for a generic NN architecture, as it is random at initialization and it changes during training.

However, in the infinite width limit, the NTK converges to a deterministic constant kernel that, simirlary to the NNGP kernel introduced in {doc}`nngp`, that can be calculated recursively:

$$
\Theta(t) \to K \mathbb{1}_{k},
$$

where $K$ is a scalar kernel calculated as:

$$
\begin{cases}
K^0(x,x') = \sigma_w^2 \frac{x \cdot x'}{n_0} + \sigma_b^2 \\
K^l(x,x') = \sigma_w^2 C^l(x,x') + K^{l-1}(x,x') \dot{C}^l(x,x') 
\end{cases}
$$

The quantity $C^l$ is the same as in the NNGP kernel, but in this case we also have another quantity, $\dot{C}^l$. They are given by:

$$
C^l(x,x') &= \mathbb{E}_{u,u'\sim\mathcal{N}(0,K^{l-1})}[\sigma(u)\sigma(u')] \\
\dot{C}^l(x,x') &= \mathbb{E}_{u,u'\sim\mathcal{N}(0,K^{l-1})}[\sigma '(u)\sigma '(u')]
$$

See for example {cite}`jacot2020neuraltangentkernelconvergence` as a reference for more details and proofs.

As a matter of facts, in the infinite width limit the NTK becomes deterministic and (for simple loss functions, as a MSE loss function) we are able to exactly solve the ODE giving the evolution of the NN outputs!

Note the similarity between the NNGP kernel and the NTK. They both arise in the infinite width limit, however they describe different situations: the former allows exact bayesian inference from the posterior while the latter describe the gradient descent dynamics and it does not correspond to generating samples from a statistical model. 
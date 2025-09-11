# Conventions and notation

As in any supervised context, the dataset is made up of ordered pairs or input data and labels: $\mathcal{D}=\{(x^\mu, y^\mu)\}_{\mu=1}^P$.
$P\in\mathbb{N}$ is the size of the dataset. $x\in\mathbb{R}^N$ (or $N_0$ for neural networks) and $y\in\mathbb{R}^D$. We will assume $D=1$ (scalar output), as the extension to the multiple output case introduces more a notational difficulty than a conceptual one.

Einstein summation convention is adopted over the components of multi-dimensional variables (lower and Latin indexes), while, unless otherwise specified, the explicit sum is written over the indexes running over data (upper and Greek indexes).

Occasionally, the matrix notation is used. For example, the dataset can be denoted also by $\mathcal{D}=\{(X, Y)\}$, where $X$ and $Y$ are the input and output data matrices, respectively. We adopt the following convention:

\begin{align*}
X&:=(x^1,\dots,x^P)\in\mathbb{R}^N\times\mathbb{R}^P, & Y&:=(y^1,\dots,y^P)^T\in\mathbb{R}^P\times\mathbb{R}^D.
\end{align*}

So that in the $D=1$ case we consider, $Y$ is simply a $P$-dimensional (column) vector, $Y\in\mathbb{R}^P$.

Every statistical model is defined specifying a _data generating model_ $y=f_\theta(x)+\varepsilon$, with $\varepsilon\sim\mathcal{N}(0,T)$ (Gaussian noise). This means that the data are extracted as

\begin{equation*}
    y^\mu \mid \theta, x^\mu \overset{iid}{\sim} \mathcal{N}(f_\theta(x^\mu), \beta^{-1}) \qquad \forall \mu=1, \dots, P
\end{equation*}

The parameter $T=\beta^{-1}$ quantifies the amount of noise, and it is denoted as the temperature parameter for its statistical physics interpretation. The noiseless case corresponds to the zero-temperature limit. The (parametric) function $f_\theta(x)$ specifies the _architecture_ and $\theta$ collectively denotes its parameters.

From the data generating model it is possible to calculate the _likelihood function_:

\begin{equation*}
    Y \mid \theta, X \sim \mathcal{N}_P(f_\theta(X), \beta^{-1}\mathbb{I}_P)
\end{equation*}

The final ingredient to fully specified a (bayesian) statistical model is the choice of a _prior_ over the parameter of the architecture, $\theta \sim p(\theta)$.
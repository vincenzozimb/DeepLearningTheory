# The (multivariate) Linear Regression model

We consider the general case of the multivariate linear regression model.

The architecture is given by:

$$
\begin{aligned}
    f_\theta(x) &: \mathbb{R}^n \to \mathbb{R}^d & f_\theta(x) &= Wx + b
\end{aligned}
$$

Where $b \in \mathbb{R}^d$ and $W \in \mathbb{R}^{d \times n}$.

The dataset is composed of $P$ examples:

$$
    \mathcal{D} = \{ (x^\mu, y^\mu) \}_{\mu=1}^P = \{ X, Y \}
$$

where:

$$
\begin{aligned}
    X &:= (x^1, \dots, x^P) \in \mathbb{R}^{n \times P} \\
    Y &:= (y^1, \dots, y^P) \in \mathbb{R}^{d \times P}
\end{aligned}
$$

In matrix notation, the architecture is therefore given by:

$$
\begin{aligned}
    f_\theta(X) &= WX + B & B &:= (b, \dots, b) \in \mathbb{R}^{d \times P}
\end{aligned}
$$

(the $B$ matrix is build by copying $P$ times the $b$ vector).
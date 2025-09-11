# Bayesian Linear Regression

Let $x\in\mathbb{R}^N$ and $y\in\mathbb{R}$. The linear regression model assumes as architecture the parametric function $f_w(x)=w^T x$, with $w\in\mathbb{R}^N$.

The statistical model has then likelihood 

$$
p(Y \mid w, X ) \propto \exp{\left\{ -\frac{\beta}{2} ||Y-X^Tw||^2 \right\}}=e^{-\beta \mathcal{L}_{MSE}(\mathcal{D}, w)},
$$

where $\mathcal{L}_{MSE}$ is the mean squared error loss function. We choose gaussian prior [^footnote1]

[^footnote1]: This choice is mathematically convenient because it is the _conjugate prior_ to the Gaussian likelihood, leading to an analytically tractable posterior.

$$
    w \sim \mathcal{N}_N(0, \Lambda^{-1}),
$$

and the posterior is then given by Bayes' theorem:

$$
    p(w \mid \mathcal{D}) 
        = \frac{p(Y \mid w, X) p(w)}{p(Y \mid X)} 
        := \frac{1}{\mathcal{Z}} \exp{\left\{ 
            -\frac{\beta}{2} (Y-X^Tw)^T(Y-X^Tw) -\frac{1}{2}w^T\Lambda w 
        \right\}}.
$$

As $w^TXY=Y^TX^Tw$ (it's a scalar quantity), we can write:

$$
    p(w \mid \mathcal{D})
        \propto \exp \left\{ 
            -\frac{1}{2}w^T (\Lambda + \beta XX^T) w + \beta Y^TX^Tw.
        \right\}
$$

Completing the square we get the posterior distribution:

\begin{align*}
    w \mid \mathcal{D} &\sim \mathcal{N}_N(\mu, M^{-1}), & 
    \begin{cases}
        M = \Lambda + \beta XX^T \\
        \mu = \beta M^{-1}XY
    \end{cases}
    .
\end{align*}

The posterior predictive is the distribution of a new (_i.e._ unseen) test example $\hat{y}$ given the dataset. Starting from:

$$
    w \mid \mathcal{D} \sim \mathcal{N}_N(\mu, M^{-1}) 
    \quad \Longrightarrow \quad
    \hat{x}^Tw \mid \hat{x}, \mathcal{D} \sim \mathcal{N}(\hat{x}^T\mu, \hat{x}^T M^{-1} \hat{x}),
$$

and using $\hat{y}=\hat{x}^Tw+\varepsilon$, where both terms are Gaussian, it follows that:

$$
    \hat{y} \mid \hat{x}, \mathcal{D} \sim \mathcal{N}(\hat{x}^T\mu, \beta^{-1} + \hat{x}^T M^{-1} \hat{x})
$$

The only missing quantity is the marginal likelihood (or bayesian evidence) $p(Y \mid X)$, closely related to the partition function of the model. 

From its definition:

$$
    p(Y \mid X) = \int d^N w \, p(Y \mid w, X) p(w) = \int d^N w \, p(Y, w \mid X)
$$

it follows that we can calculate it as the marginal of the joint distribution $p(Y, w \mid X)$. This is a useful fact for this model, as we have normal distributions which are analytically tractable [^footnote2]. In particular, both $p(Y \mid w, X)$ and $p(w)$ are Gaussians, and $y$ is an affine function of $w$, and this implies that also the joint distribution is gaussian, with covariance:

[^footnote2]: Again, this is a consequence of the _conjugacy_ of the model.

\begin{equation*}
    \begin{aligned}
        \mathbb{C}ov(y^\mu, w_i) &= \mathbb{C}ov(w_jx_j^\mu + \varepsilon^\mu, w_i) = x_j^\mu (\Lambda^{-1})_{ji} \\
        \mathbb{C}ov(y^\mu, y^\nu) &= \mathbb{C}ov(w_ix_i^\mu + \varepsilon^\mu, w_jx_j^\nu + \varepsilon^\nu) = x_i^\mu x_j^\nu (\Lambda^{-1})_{ij} + \beta^{-1} \delta^{\mu\nu}
    \end{aligned}
\end{equation*}

So that:

\begin{equation*}
    w, Y \mid X \sim \mathcal{N}_{N+P}\left(0, 
    \begin{pmatrix}
        \Lambda^{-1}    & \Lambda^{-1T}X \\
        X^T\Lambda^{-1} & X^T\Lambda^{-1}X + \beta^{-1}\mathbb{I}_P
    \end{pmatrix}
    \right).
\end{equation*}

Therefore, the marginal is simply:

$$
    Y \mid X \sim \mathcal{N}_P(0, X^T\Lambda^{-1}X + \beta^{-1}\mathbb{I}_P)
$$ (eq:MarginalLikelihood-BayesianLinearRegression)

and the log-partition function is (up to an additive constant)

$$
    \ln \mathcal{Z} = -\frac{1}{2} \ln \det (X^T\Lambda^{-1}X + \beta^{-1}\mathbb{I}_P) -\frac{1}{2} Y^T (X^T\Lambda^{-1}X + \beta^{-1}\mathbb{I}_P)^{-1} Y.
$$

Another way to derive this result is to use $Y=X^Tw + \varepsilon\mathbb{I}_P$ and the fact that both terms are normal distributed to directly obtain {eq}`eq:MarginalLikelihood-BayesianLinearRegression`.
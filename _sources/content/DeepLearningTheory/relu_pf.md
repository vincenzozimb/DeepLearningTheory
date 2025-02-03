# Extention for finite mean activation functions

See the page {doc}`pf`.

We can restart the calculation from:

$$
    \mathcal{Z} = 
    \int \prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} 
    \exp \left[
        - \frac{\beta}{2} \sum_\mu (y^\mu - s^\mu)^2 
        + i \sum_\mu s^\mu \bar{s}^\mu 
    \right] \times 
    \left[ 
        \int dq ~ e^{-q^2/2} P(q) 
    \right]^{N_1}
$$

In the general case:

$$
q = \frac{1}{\sqrt{\lambda_1N_1}}\sum_\mu \bar{s}^\mu \sigma(h^\mu).
$$

With mean value:

$$
    \mathbb{E}[q] = 
    \frac{1}{\sqrt{\lambda_1N_1}} \sum_\mu \bar{s}^\mu \mathbb{E}[\sigma(h^\mu)] =: 
    \frac{1}{\sqrt{\lambda_1N_1}} \sum_\mu \bar{s}^\mu m^\mu =:
    m_q(\bar{s}, C).
$$

So, applying the BM theorem to the variable $q - m_q(\bar{s}, C)$ yields $q \sim \mathcal{N}(m_q(\bar{s}, C), Q(\bar{s}, C))$, where, this time:

$$
Q(\bar{s}, C) := \frac{1}{\lambda_1N_1} \sum_{\mu\nu} \bar{s}^\mu (K_{\mu\nu} -m_\mu m_\nu) \bar{s}^\nu
$$

So the integral in the $q$ variable is now (in this calculation I omit the dependence of $Q$, $Q=Q(\bar{s}, C)$):

```{admonition} Formula
:class: dropdown tip

Gaussian integral

$$
    \int_{-\infty}^{\infty} dx ~ e^{-(ax^2+bx+c)} = \sqrt{\frac{\pi}{a}} e^{\frac{b^2}{4a}-c}
$$

```

$$
    \frac{1}{\sqrt{2\pi Q}} \int dq ~ e^{-(q - m_q)^2 / 2Q} e^{-q^2/2} =
    \frac{1}{\sqrt{1+Q}} \exp \left( 
        -\frac{m_q^2}{2(1+Q)} 
    \right)
$$


Thus:

$$
    \mathcal{Z} &= 
    \int \prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \exp \left[
        - \frac{\beta}{2} \sum_\mu (y^\mu - s^\mu)^2 
        + i \sum_\mu s^\mu \bar{s}^\mu 
    \right] \times \\
    
    & \qquad \times 
    (1+Q(\bar{s},C))^{-N_1/2} \exp \left[
        - \frac{N_1 m_q(\bar{s}, C)^2 / 2}{1+Q(\bar{s}, C)}
    \right].
$$


Let's solve the integral in the $s$ variables to simplify the expression:

\begin{align*}

    \mathcal{Z} &\propto
    \beta^{-P/2} \int \prod_\mu d\bar{s}^\mu \exp \left[
        + i \sum_\mu y^\mu \bar{s}^\mu 
        - \frac{1}{2\beta} \sum_\mu (s^\mu)^2 
    \right] \times \\

    & \qquad \times 
    (1+Q(\bar{s}, C))^{-N_1/2} \exp \left[
        - \frac{N_1 m_q^2(\bar{s}, C) / 2}{1+Q(\bar{s}, C)}
    \right].

\end{align*}


Inserting a Dirac delta over $Q$ and its Fourier representation:

\begin{align*}

    \mathcal{Z} &\propto
    \beta^{-P/2} \int dQ d\bar{Q} \prod_\mu d\bar{s}^\mu \exp \left[
        + i \sum_\mu y^\mu \bar{s}^\mu 
        - \frac{1}{2\beta} \sum_\mu (s^\mu)^2 
    \right] \times \\

    & \qquad \times 
    (1+Q)^{-N_1/2} \exp \left[
        - \frac{N_1 m_q^2(\bar{s}, C) / 2}{1+Q}
    \right] \exp [iQ\bar{Q} - i \bar{Q}Q(\bar{s}, C)].

\end{align*}


Changing $\bar{Q} \to -iN_1\bar{Q}/2$:

\begin{align*}

    \mathcal{Z} &\propto 
    \frac{N_1}{2} \beta^{-P/2} \int dQ d\bar{Q} 
    (1+Q)^{-N_1/2} \exp \left[
        \frac{N_1}{2}Q\bar{Q} 
        + i \sum_\mu y^\mu \bar{s}^\mu
        - \frac{1}{2\beta} \sum_\mu (s^\mu)^2
    \right] \times \\

    & \qquad \times \exp \left[
        - \frac{1}{1+Q}\frac{1}{2\lambda_1} \sum_{\mu\nu} \bar{s}^\mu m_\mu m_\nu \bar{s}^\nu 
        - \frac{\bar{Q}}{2\lambda_1} \sum_{\mu\nu} \bar{s}^\mu (K_{\mu\nu} 
        - m_\mu m_\nu) \bar{s}^\nu
    \right]

\end{align*}


This is the same integral of the zero-mean case, upon substituting:

\begin{align*}
    \frac{\bar{Q}}{\lambda_1} K &\to \frac{\bar{Q}}{\lambda_1} K - \frac{\bar{Q}}{\lambda_1} m \otimes m + \frac{1}{\lambda_1(1+Q)} m \otimes m \\
    
    & \to \frac{\bar{Q}}{\lambda_1} K - \frac{1}{\lambda_1} \left( \bar{Q} - \frac{1}{1+Q} \right) m \otimes m.
\end{align*}

# Extention for finite mean activation functions

See the page {doc}`pf`.

We can restart the calculation from:

$$
\mathcal{Z} = A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] \times \left[ \int dq ~ e^{-q^2/2} P(q) \right]^{N_1}
$$

In the general case:

$$
q = \frac{1}{\sqrt{\lambda_1N_1}}\sum_\mu \bar{s}^\mu \sigma(h^\mu).
$$

With mean value:

$$
\mathbb{E}[q] = \frac{1}{\sqrt{\lambda_1N_1}}\sum_\mu \bar{s}^\mu \mathbb{E}[\sigma(h^\mu)] =: \frac{1}{\sqrt{\lambda_1N_1}}\sum_\mu \bar{s}^\mu m^\mu=:\langle q \rangle.
$$

So, applying the BM theorem to the variable $q - \langle q \rangle$ yields $q \sim \mathcal{N}(\langle q \rangle, Q(\bar{s}, C))$, where again:

$$
Q(\bar{s}, C) := \frac{1}{\lambda_1N_1} \sum_{\mu\nu}\bar{s}^\mu K_{\mu\nu} \bar{s}^\nu
$$

So the integral in the $q$ variable is now (in this calculation I omit the dependence of $Q$):

```{admonition} Formula
:class: dropdown tip

Gaussian integral

$$
\int_{-\infty}^{\infty} dx ~ e^{-(ax^2+bx+c)} = \sqrt{\frac{\pi}{a}} e^{\frac{b^2}{4a}-c}
$$

```

$$
\frac{1}{\sqrt{2\pi Q}} \int dq ~ e^{-(q - \langle q \rangle)^2 / 2Q} e^{-q^2/2} = \frac{1}{\sqrt{1+Q}} \exp \left( -\frac{\langle q \rangle^2}{2(1+Q)} \right)
$$

The $\langle q \rangle^2$ term is:

\begin{align*}
\langle q \rangle^2 &= \frac{1}{\lambda_1N_1} \sum_{\mu\nu} \bar{s}^\mu K^{(1)}_{\mu\nu} \bar{s}^\nu, & K^{(1)}_{\mu\nu}&:=m^\mu m^\nu.
\end{align*}

So:

$$
\mathcal{Z} &= A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] (1+Q(\bar{s},C))^{-N_1/2} \times \\
&\qquad \times \exp \left[ -\frac{1}{2\lambda_1(1+Q(\bar{s}, C))}\sum_{\mu\nu}\bar{s}^\mu K^{(1)}_{\mu\nu} \bar{s}^\nu \right].
$$

Inserting a Dirac delta over $Q$ and its Fourier representation:

$$
\mathcal{Z} &= A \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \frac{dQd\bar{Q}}{2\pi} \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] (1+Q)^{-N_1/2} \times \\
&\qquad \times e^{iQ\bar{Q}} \exp \left[ -\frac{1}{2\lambda_1(1+Q)}\sum_{\mu\nu}\bar{s}^\mu K^{(1)}_{\mu\nu} \bar{s}^\nu -i\frac{\bar{Q}}{\lambda_1N_1} \sum_{\mu\nu}\bar{s}^\mu K_{\mu\nu} \bar{s}^\nu \right].
$$

Changing $\bar{Q} \to -iN_1\bar{Q}/2$:

$$
\mathcal{Z} &= A \frac{N_1}{2} \int\left(\prod_\mu \frac{ds^\mu d\bar{s}^\mu}{2\pi} \right) \frac{dQd\bar{Q}}{2\pi} \exp \left[-\frac{\beta}{2}\sum_\mu (y^\mu - s^\mu)^2 +i \sum_\mu s^\mu \bar{s}^\mu \right] (1+Q)^{-N_1/2} \times \\
&\qquad \times e^{\frac{N_1}{2}Q\bar{Q}} \exp \left[ -\frac{1}{2\lambda_1(1+Q)}\sum_{\mu\nu}\bar{s}^\mu K^{(1)}_{\mu\nu} \bar{s}^\nu -\frac{\bar{Q}}{2\lambda_1} \sum_{\mu\nu}\bar{s}^\mu K_{\mu\nu} \bar{s}^\nu \right].
$$

The same integral of the zero-mean case is obtained under the substitution:

$$
\frac{\bar{Q}}{\lambda_1} K \to \frac{\bar{Q}}{\lambda_1} K + \frac{1}{\lambda_1(1+Q)}K^{(1)}.
$$

```{warning}
In the final result there is a factor $-\frac{\bar{Q}}{\lambda_1}K^{(1)}$ that is missing in my calculation.
```

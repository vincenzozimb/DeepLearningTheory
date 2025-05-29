# Single vatiable expression for $S$ in the case of zero-mean activation functions

See the page {doc}`pf`. 

It turns out that, at least in the case of zero-mean activation functions, it is possible to derive an effective action that depends only on a single variable. In particular, it is not necessary to introduce the variables $Q$ and its conjugate $\bar{Q}$ as done in {doc}`pf`.

The single variable expression follows from an integral identity for gamma distribution (in the single hidden layer case), and in particular using the moment generating function / Laplace transform of a gamma distribution [^mylabel]. The integral equality is the following:

[^mylabel]: Or equivalently in this simple case, of a $\chi^2$ distribution.


\begin{equation}
    (1 + Q(\bar{s}, C))^{-N_1/2} =
    \frac{1}{\Gamma(N_1/2)} \int_{0}^{\infty} dQ
    Q^{\frac{N_1}{2}-1} e^{-Q(1+Q(\bar{s}, C))}. 
\end{equation}

In the multilayer case, a generalization of this idendity can be found using Wishart distributions.

We can restart the calculation from:

$$

    \mathcal{Z} \propto \beta^{-P/2} \int \prod_\mu d\bar{s}^\mu
    \exp \left\{
        i \bar{s}^t \cdot y + \frac{1}{2\beta}||\bar{s}||^2
    \right\} \left(
        1 + \frac{1}{\lambda_1 N_1} \bar{s}^t \cdot K \cdot \bar{s}
    \right)^{-N_1/2}.

$$


Using the integral idendity and changing $Q \to N_1 Q / 2$ yields:

$$
    \mathcal{Z} \propto \beta^{-P/2} \frac{N_1/2}{\Gamma(N_1/2)} 
    \left(\frac{N_1}{2}\right)^{N_1/2}
    \int_{0}^{\infty} dQ Q^{N_1/2} 
$$


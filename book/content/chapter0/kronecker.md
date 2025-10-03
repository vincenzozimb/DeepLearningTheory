# Technicalities: Kronecker product of matrices and the vectorization operation

If $A \in \mathbb{R}^{n \times p}$ and $B \in \mathbb{R}^{m \times q}$, the **Kronecker product** of $A$ and $B$ is the block-matrix defined by:

$$
\mathbb{R}^{nm \times pq} \ni A \otimes B = \{ A_{ij} B \}_{ij} = 
\begin{bmatrix} 
    A_{11}B & \dots     & A_{1p}B \\
    \vdots  & \ddots    & \\
    A_{n1}B &           & A_{np}B 
\end{bmatrix}
$$

The **vectorization** of a matrix is obtained by stacking the columns (for vectors, I always mean *column* vectors)

$$
\mathbb{R}^{np} \ni Vec(A) :=
\begin{bmatrix} 
    A_{1}   \\
    \vdots  \\
    A_{p} 
\end{bmatrix}
$$

where $A_i \in \mathbb{R}^n$ is the $i$-th column of A.

## Properties

1. $A \otimes (\cdot)$ is a linear operator on matrices (easy to prove):

$$
A \otimes (c_1 B_1 + c_2 B_2) = c_1 A \otimes B_1 + c_2 A \otimes B_2
$$


2. Associativity (not so easy to prove):

$$
A \otimes (B \otimes C) = (A \otimes B) \otimes C
$$


3. Mixed product (not so difficult to prove). For matrices of compatible dimensions:

$$
(A \otimes B) (C \otimes D) = (AC) \otimes (BD)
$$

Pay attention to the dimension. This is the only possible result.


4. Transpose (easy to prove):

$$
(A \otimes B)^\top = A^\top \otimes B^\top
$$


5. Inverse (easy to prove using (3))

$$
(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}
$$


6. Left matrix is a block matrix (easy to prove):

$$
A =
\begin{bmatrix} 
    \alpha & \beta  \\
    \gamma & \delta
\end{bmatrix} 
\Longrightarrow
(A \otimes B) =
\begin{bmatrix} 
    \alpha \otimes B & \beta \otimes B \\
    \gamma \otimes B & \delta \otimes B
\end{bmatrix} 
$$


7. Trace (easy to prove):

$$
\Tr(A \otimes B) = \Tr(A) \Tr(B)
$$


8. Determinant:

$$
A \in \mathbb{R}^{n \times n}, B \in \mathbb{R}^{m \times m} \Longrightarrow \det(A \otimes B) = (\det A)^m (\det B)^n
$$


9. Mixed Kronecker matrix - vector product, or "**vec trick**" (proof later):

$$
Vec(AMB^\top) = (B \otimes A) Vec(M)
$$


10. Vectorization of a rank-one matrix (easy to prove):

$$
a \in \mathbb{R}^n, b \in \mathbb{R}^m, ab^\top \in \mathbb{R}^{n \times m}
\Longrightarrow
Vec(a b^\top) = b \otimes a 
$$


## Proof of the "vec trick"

To complete the proof, we need two facts:

- The vectorization operation is a _linear_ operation

- For every matrix M, it is always possible to write it as:

$$
M = \sum_{ij} M_{ij} e_i e_j^\top
$$

where $e_i, e_j$ are the columns of the idendity matrices (of proper dimensions, here with a little abuse of notation I use the same symbol).


The proof procedes as follows:

$$
\begin{aligned}
    Vec(AMB^\top)   &= Vec(A \sum_{ij} M_{ij} e_i e_j^\top B^\top) \\
                    &= \sum_{ij} M_{ij} Vec(A e_i e_j^\top B^\top) \\
                    &= \sum_{ij} M_{ij} Vec( (A e_i) (B e_j)^\top ) \\
                    (10) &= \sum_{ij} M_{ij} (B e_j) (A e_i) \\
                    (3) &= \sum_{ij} M_{ij} (B \otimes A) (e_j \otimes e_i) \\
                    &= (B \otimes A) \sum_{ij} M_{ij} (e_j \otimes e_i) \\
                    &= (B \otimes A) \sum_{ij} M_{ij} Vec(e_i e_j^\top) \\
                    &= (B \otimes A) Vec(\sum_{ij} M_{ij} e_i e_j^\top) \\
                    &= (B \otimes A) Vec(M) \\
\end{aligned}
$$

And the proof is completed $\square$
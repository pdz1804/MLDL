---
layout: post
title: 04. Data Dimension Greater than the Number of Data Points
mathjax: true
tags:
- Basic_Math
- Linear_Algebra
categories: Basic_Math_and_Linear_Algebra
description: Details information about the Data Dimension Greater than the Number of Data Points
---

## Data Dimension Greater than the Number of Data Points

This occurs when $D > N$, meaning the data matrix $\mathbf{X}$ is a
'tall matrix'. In this case, the number of non-zero eigenvalues of the
covariance matrix $\mathbf{S}$ does not exceed its rank, i.e., $N$.
Thus, we must choose $K \leq N$ because it is impossible to select
$K > N$ non-zero eigenvalues of a matrix with rank $N$.

Eigenvalue and eigenvector computation can be efficiently performed
based on the following properties:

**Property 1**: An eigenvalue of $\mathbf{A}$ is also an eigenvalue of
$k\mathbf{A}$ for any non-zero $k$. This can be directly deduced from
the definitions of eigenvalues and eigenvectors.

**Property 2**: The eigenvalues of $\mathbf{AB}$ are also eigenvalues of
$\mathbf{BA}$ where $\mathbf{A} \in \mathbb{R}^{d_1 \times d_2}$ and
$\mathbf{B} \in \mathbb{R}^{d_2 \times d_1}$ are arbitrary matrices with
non-zero dimensions $d_1, d_2$.

Thus, instead of finding the eigenvalues of the covariance matrix
$\mathbf{S} \in \mathbb{R}^{D\times D}$, we find the eigenvalues of the
smaller matrix
$\mathbf{T} = \mathbf{X}^T \mathbf{X} \in \mathbb{R}^{N \times N}$
(since $N < D$).

**Property 3:** If $(\lambda, \mathbf{u})$ is an eigenvalue-eigenvector
pair of $\mathbf{T}$, then $(\lambda, \mathbf{Xu})$ is an
eigenvalue-eigenvector pair of $\mathbf{S}$.

Indeed: $$\begin{aligned}
  \mathbf{X}^T \mathbf{Xu} &=& \lambda \mathbf{u} \quad (7) \\
  \Rightarrow (\mathbf{X}\mathbf{X}^T)(\mathbf{Xu}) &=& \lambda \mathbf{Xu} \quad (8)
\end{aligned}$$

Expression $(7)$ follows from the eigenvalue definition. Expression
$(8)$ is derived by multiplying both sides of $(7)$ by $\mathbf{X}$ on
the left. This leads to **Observation 3**.

Thus, we can compute the eigenvalues and eigenvectors of the covariance
matrix using a smaller matrix.

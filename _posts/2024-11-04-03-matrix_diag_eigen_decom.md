---
layout: post
title: 03. Matrix Diagonalization and Eigen Decomposition
mathjax: true
tags:
- Basic_Math
- Linear_Algebra
categories: Basic_Math_and_Linear_Algebra
description: Details information about the Matrix Diagonalization and Eigen Decomposition
---

## Matrix Diagonalization and Eigen Decomposition

You may remember a common problem in Linear Algebra: Matrix Diagonalization. This problem states that a square matrix \\(\mathbf{A} \in \mathbb{R}^{n\times n}\\) is called *diagonalizable* if there exists a diagonal matrix \\(\mathbf{D}\\) and an invertible matrix \\(\mathbf{P}\\) such that:
\\[\mathbf{A} = \mathbf{P} \mathbf{D} \mathbf{P}^{-1}~~~~(1)\\]
The number of non-zero elements in the diagonal matrix \\(\mathbf{D}\\) represents the rank of matrix \\(\mathbf{A}\\).

Multiplying both sides of \\((1)\\) by \\(\mathbf{P}\\) gives:
\\[\mathbf{AP} = \mathbf{PD}~~~~(2)\\]

Let \\(\mathbf{p} _ {i}, \mathbf{d} _ {i}\\) denote the i-th column of matrices \\(\mathbf{P}\\) and \\(\mathbf{D}\\), respectively. Since each column on both sides of \\((2)\\) must be equal, we have:
\\[\mathbf{Ap} _ i = \mathbf{Pd} _ i = d_{ii}\mathbf{p} _ i ~~~~ (3)\\]
where \\(d _ {ii}\\) is the i-th element of \\(\mathbf{D}\\).

The second equality arises because \\(\mathbf{D}\\) is a diagonal matrix, meaning \\(\mathbf{d} _ i\\) only has \\(d _ {ii}\\) as a non-zero component. This expression \\((3)\\) shows that each \\(d _ {ii}\\) is an *eigenvalue* of \\(\mathbf{A}\\), and each column vector \\(\mathbf{p} _ i\\) is an *eigenvector* of \\(\mathbf{A}\\) corresponding to the eigenvalue \\(d_ {ii}\\).

The factorization of a square matrix as in \\((1)\\) is known as *Eigen Decomposition*.

A key point is that the decomposition \\((1)\\) only applies to square matrices and may not always exist. It only exists if \\(\mathbf{A}\\) has \\(n\\) linearly independent eigenvectors; otherwise, an invertible \\(\mathbf{P}\\) does not exist. Additionally, this decomposition is not unique, as if \\(\mathbf{P}, \mathbf{D}\\) satisfy \\((1)\\), then \\(k\mathbf{P}, \mathbf{D}\\) also satisfy it for any non-zero real \\(k\\).

Decomposing a matrix into products of special matrices (Matrix Factorization or Matrix Decomposition) has significant benefits, such as dimensionality reduction, data compression, exploring data characteristics, solving linear equations, clustering, and many other applications. Recommendation Systems are one of the many applications of Matrix Factorization.

In this article, I will introduce a beautiful Matrix Factorization method in Linear Algebra: Singular Value Decomposition (SVD). You will see that any matrix, not necessarily square, can be decomposed into the product of three special matrices.

### Eigenvalues and Eigenvectors

For a square matrix \\(\mathbf{A} \in \mathbb{R}^{n\times n}\\), a scalar \\(\lambda\\) and a non-zero vector \\(\mathbf{x} \in \mathbb{R}^n\\) are an eigenvalue and eigenvector, respectively, if:
\\[\mathbf{Ax} = \lambda \mathbf{x}\\]

**A few properties:**

1. If \\(\mathbf{x}\\) is an eigenvector of \\(\mathbf{A}\\) associated with \\(\lambda\\), then \\(k\mathbf{x}, k \neq 0\\) is also an eigenvector associated with \\(\lambda\\).

2. Any square matrix of order \\(n\\) has \\(n\\) eigenvalues (including repetitions), which may be complex.

3. For symmetric matrices, all eigenvalues are real.

4. For a positive definite matrix, all its eigenvalues are positive real numbers. For a positive semi-definite matrix, all its eigenvalues are non-negative real numbers.

The last property can be derived from the definition of a (semi-)positive definite matrix. Indeed, let \\(\mathbf{u} \neq \mathbf{0}\\) be an eigenvector of a positive semi-definite matrix \\(\mathbf{A}\\) with eigenvalue \\(\lambda\\), then:
\\[\mathbf{Au} = \lambda \mathbf{u} \Rightarrow \mathbf{u}^T\mathbf{Au} = \lambda \mathbf{u}^T\mathbf{u} = \lambda \|\mathbf{u}\|_2^2\\]

Since \\(\mathbf{A}\\) is positive semi-definite, \\(\mathbf{u}^T\mathbf{Au} \geq 0\\) for all \\(\mathbf{u} \neq \mathbf{0}\\), implying that \\(\lambda\\) is non-negative.

### Orthogonal and Orthonormal Bases

A basis \\(\{\mathbf{u}_1, \mathbf{u}_2,\dots, \mathbf{u}_m \in \mathbb{R}^m\}\\) is called *orthogonal* if each vector is non-zero and any two different vectors are orthogonal:
\\[\mathbf{u}_i \neq \mathbf{0}; ~~ \mathbf{u}_i^T \mathbf{u}_j = 0 ~ \forall ~1 \leq i \neq j \leq m\\]

An *orthonormal* basis \\(\{\mathbf{u}_1, \mathbf{u}_2,\dots, \mathbf{u}_m \in \mathbb{R}^m\}\\) is an orthogonal basis in which each vector has a Euclidean length (2-norm) of 1:

\\[ \mathbf{u}_i^T \mathbf{u}_j = 1 \\ \text{if} \\ i = j \\]
\\[ \mathbf{u}_i^T \mathbf{u}_j = 0 \\ \text{otherwise} \\]

Let \\(\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2,\dots, \mathbf{u}_m]\\), where \\(\{\mathbf{u}_1, \mathbf{u}_2,\dots, \mathbf{u}_m \in \mathbb{R}^m\}\\) is orthonormal. Then, from \\((4)\\), we have:
\\[\mathbf{UU}^T = \mathbf{U}^T\mathbf{U} = \mathbf{I}\\]
where \\(\mathbf{I}\\) is the \\(m\\)-order identity matrix. We call \\(\mathbf{U}\\) an *orthogonal matrix*.

**Some properties:**

1. \\(\mathbf{U}^{-1} = \mathbf{U}^T\\): the inverse of an orthogonal matrix is its transpose.

2. If \\(\mathbf{U}\\) is orthogonal, then \\(\mathbf{U}^T\\) is also orthogonal.

3. The determinant of an orthogonal matrix is \\(1\\) or \\(-1\\).

4. An orthogonal matrix represents a *rotation* of a vector. If we rotate vectors \\(\mathbf{x},\mathbf{y} \in \mathbb{R}^m\\) with an orthogonal matrix \\(\mathbf{U} \in \mathbb{R}^{m \times m}\\), the inner product of the rotated vectors remains unchanged:
\\[(\mathbf{Ux})^T (\mathbf{Uy}) = \mathbf{x}^T \mathbf{U}^T \mathbf{Uy} = \mathbf{x}^T\mathbf{y}\\]

1. Let \\(\hat{\mathbf{U}} \in \mathbb{R}^{m \times r}, r < m\\) be a submatrix of \\(\mathbf{U}\\) formed by \\(r\\) columns of \\(\mathbf{U}\\), then \\(\hat{\mathbf{U}}^T\hat{\mathbf{U}} = \mathbf{I}_{r}\\)
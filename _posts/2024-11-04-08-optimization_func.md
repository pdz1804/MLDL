---
layout: post
title: 08. Optimization Problems
mathjax: true
tags:
- Basic_Math
- Linear_Algebra
categories: Basic_Math_and_Linear_Algebra
description: Details information about the Optimization Problems
---

## Optimization Problems

In Optimization, a constrained optimization problem is often written in
the following form: 

$$\begin{aligned}
    \mathbf{x}^* &=& \arg\min_{\mathbf{x}} f_0(\mathbf{x}) \nonumber \\
    \text{subject to: } && f_i(\mathbf{x}) \leq 0, \quad i = 1, 2, \dots, m \nonumber \\
    && h_j(\mathbf{x}) = 0, \quad j = 1, 2, \dots, p \nonumber
\end{aligned}$$

Here, the vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$ is called the
*optimization variable*. The function
$f_0: \mathbb{R}^n \rightarrow \mathbb{R}$ is referred to as the
*objective function* (in Machine Learning, objective functions are often
referred to as loss functions). The functions
$f_i, h_j: \mathbb{R}^n \rightarrow \mathbb{R}, i = 1, 2, \dots, m; j = 1, 2, \dots, p$
are the *constraint functions* (or simply constraints). The set of
points $\mathbf{x}$ that satisfy the constraints is called the *feasible
set*. Any point in the feasible set is referred to as a *feasible
point*, while those not in the feasible set are called *infeasible
points*.

**Notes:**

-   If the problem is to find the maximum instead of the minimum, we
    simply negate $f_0(\mathbf{x})$.

-   If the constraint is \"$\geq$\", i.e., $f_i(\mathbf{x}) \geq b_i$,
    we can negate the constraint to get \"$\leq$\" by writing
    $-f_i(\mathbf{x}) \leq -b_i$.

-   Constraints can also be strict inequalities.

-   An equality constraint, i.e., $h_j(\mathbf{x}) = 0$, can be written
    as two inequalities $h_j(\mathbf{x}) \leq 0$ and
    $-h_j(\mathbf{x}) \leq 0$. In some texts, equality constraints are
    omitted.

-   In this text, $\mathbf{x}, \mathbf{y}$ are mainly used to denote
    variables, not data as in previous discussions. The optimization
    variable is the one shown in the $\arg\min$ notation.

In general, optimization problems do not have a universal solution
method, and some problems are still unsolved. Most solution methods
cannot guarantee that the solution is the global optimum, i.e., the true
minimum or maximum. Instead, the solution is often a local optimum,
i.e., a local extremum.

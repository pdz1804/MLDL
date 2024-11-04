---
layout: post
title: 19. Likelihood Function and Maximum Likelihood Estimation
mathjax: true
tags:
- Advanced Machine Learning
- Basic Math and Linear Algebra
categories: Advanced_Machine_Learning
description: Details information about the Likelihood Function and Maximum Likelihood Estimation
---

## Likelihood Function and Maximum Likelihood Estimation

### Likelihood Function

In logistic regression, **the likelihood function represents the
probability of observing the given data under a specific set of model
parameters**. It is defined as the joint probability of all observed
outcomes (binary labels) given their corresponding predicted
probabilities.

For logistic regression, the likelihood of a single observation $i$ is:

$$L_i = P(y_i | X_i)^{y_i} \times (1 - P(y_i | X_i))^{1 - y_i}$$

where $y_i \in \{0, 1\}$ is the observed class label for the $i$-th data
point, and $P(y_i | X_i)$ is the predicted probability of class 1.

The **total likelihood** for all observations is then the product of
each individual likelihood:

$$L(\beta) = \prod_{i=1}^{N} L_i$$


### Log-Likelihood Function

Directly maximizing the likelihood function (as is done in Maximum
Likelihood Estimation) can be computationally challenging due to the
multiplicative terms in the product. To simplify calculations, we often
use the log-likelihood, which is the natural logarithm of the likelihood
function. Since the logarithm is a monotonic function, maximizing the
log-likelihood is equivalent to maximizing the likelihood.

The log-likelihood function for logistic regression is:

$$\log L(\beta) = \sum_{i=1}^{N} \left[ y_i \log(P(y_i | X_i)) + (1 - y_i) \log(1 - P(y_i | X_i)) \right]$$

This transformation turns the product into a sum, simplifying the
process of optimization.


### Maximum Likelihood Estimation

Maximum Likelihood Estimation is a method to estimate the parameters of
a statistical model by maximizing the likelihood function. In logistic
regression, Maximum Likelihood Estimation aims to find the values of
$\beta$ that maximize the likelihood of observing the data. Essentially,
Maximum Likelihood Estimation seeks the model parameters that make the
observed data \"most probable\" under the logistic model.

The reason for using Maximum Likelihood Estimation is rooted in
probability theory: by maximizing the likelihood, we obtain the
parameter values that are most likely to have generated the observed
data. This approach has strong theoretical foundations because:

-   Maximum Likelihood Estimation estimators are often **consistent**,
    meaning they converge to the true parameter values as the sample
    size increases.

-   They are also **efficient** and **asymptotically unbiased**,
    providing accurate estimates with large enough data.


### Optimization in Logistic Regression

Maximizing the log-likelihood in logistic regression typically requires
iterative optimization methods because the function is nonlinear and
does not have a closed-form solution.

During each iteration, these algorithms adjust the model parameters
(coefficients) to increase the log-likelihood until convergence. Once
the log-likelihood reaches a maximum, the resulting parameter estimates
are considered the Maximum Likelihood Estimates of the model
coefficients.

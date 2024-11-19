---
layout: post
title: 07. Gradient Descent for Multi-Variable Functions
mathjax: true
tags:
- Basic_Math
- Linear_Algebra
categories: Basic_Math_and_Linear_Algebra
description: Details information about the Gradient Descent for Multi-Variable Functions
---

## Gradient Descent for Multi-variable Functions {#sec:gd-multi-variable}

Suppose we need to find the global minimum for the function
$f(\mathbf{\theta})$ where $\mathbf{\theta}$ (*theta*) is a vector,
often used to denote the set of parameters of a model to be optimized
(in Linear Regression, the parameters are the coefficients
$\mathbf{w}$). The derivative of the function at any point $\theta$ is
denoted as $\nabla_{\theta}f(\theta)$ (the inverted triangle is read as
*nabla*). Similar to single-variable functions, the GD algorithm for
multi-variable functions also starts with an initial guess $\theta_{0}$,
and at the $t$-th iteration, the update rule is:

$\theta_{t+1} = \theta_{t} - \eta \nabla_{\theta} f(\theta_{t})$

Or written more simply:
$\theta = \theta - \eta \nabla_{\theta} f(\theta)$.

The rule to remember: **always go in the opposite direction of the
derivative**.

Calculating derivatives of multi-variable functions is a necessary
skill. Some simple derivatives can be [found here](https://tiepvupsu.github.io/math/#bang-cac-dao-ham-co-ban).

### Back to Linear Regression {#sec:linear-regression}

In this section, we return to the [Linear Regression](https://tiepvupsu.github.io/2016/12/28/linearregression/) problem and try to optimize its loss function using the GD algorithm.

The loss function of Linear Regression is:
$\mathcal{L}(\mathbf{w}) = \dfrac{1}{2N}\|\|\mathbf{y - \bar{X}w}\|\|_2^2$

**Note**: this function is slightly different from the one I mentioned
in the [Linear Regression](https://tiepvupsu.github.io/2016/12/28/linearregression/) article. The denominator includes $N$, the number of data points in the training set. Averaging the error helps avoid cases where the loss function and derivative have very large values, affecting the accuracy of calculations on computers. Mathematically, the solutions to the two problems are the same.

The derivative of the loss function is:
$\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w}) = 
\dfrac{1}{N}\mathbf{\bar{X}}^T \mathbf{(\bar{X}w - y)} ~~~~~(1)$

### Example in Python and Some Programming Notes {#sec:python-example}

Load libraries

```python
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(2)
```

Next, we create 1000 data points chosen *close* to the line
$y = 4 + 3x$, display them, and find the solution using the formula:

```python
X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ',w_lr.T)

# Display result
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1*x0

# Draw the fitting line 
plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()
```

```python
Solution found by formula: w =  [[ 4.00305242  2.99862665]]
```

Next, we write the derivative and loss function:

```python
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2;
```

### Checking the Derivative {#sec:check-derivative}

Calculating derivatives of multi-variable functions is usually quite
complex and prone to errors. If we calculate the derivative incorrectly,
the GD algorithm cannot run correctly. In practice, there is a way to
check whether the calculated derivative is accurate. This method is
based on the definition of the derivative (for single-variable
functions):
$f'(x) = \lim_{\varepsilon \rightarrow 0}\dfrac{f(x + \varepsilon) - f(x)}{\varepsilon}$

A commonly used method is to take a very small value $\varepsilon$, for
example, $10^{-6}$, and use the formula:
$f'(x) \approx \dfrac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon} ~~~~ (2)$

This method is called the *numerical gradient*.

**Question: Why is the two-sided approximation formula above widely used, and why not use the right or left derivative approximation?**

There are two explanations for this issue, one geometric and one
analytic.

### Geometric Explanation {#sec:geometric-explanation}

Observe the image below:

In the image, the red vector is the *exact* derivative of the function
at the point with abscissa $x_0$. The blue vector (which appears
slightly purple after converting from .pdf to .png) represents the right
derivative approximation. The green vector represents the left
derivative approximation. The brown vector represents the two-sided
derivative approximation. Among these approximations, the two-sided
brown vector is closest to the red vector in terms of direction.

The difference between the approximations becomes even more significant
if the function is *bent* more strongly at point x. In that case, the
left and right approximations will differ significantly. The two-sided
approximation will be more *stable*.

### Analytic Explanation {#sec:analytic-explanation}

Let's revisit a bit of first-year university Calculus I: [Taylor Series Expansion](http://mathworld.wolfram.com/TaylorSeries.html).

With a very small $\varepsilon$, we have two approximations:

$f(x + \varepsilon) \approx f(x) + f'(x)\varepsilon + \dfrac{f"(x)}{2} \varepsilon^2 + \dots$

and:
$f(x - \varepsilon) \approx f(x) - f'(x)\varepsilon + \dfrac{f"(x)}{2} \varepsilon^2 - \dots$

From this, we have:
$\dfrac{f(x + \varepsilon) - f(x)}{\varepsilon} \approx f'(x) + \dfrac{f"(x)}{2}\varepsilon + \dots =  f'(x) + O(\varepsilon) ~~ (3)$

$\dfrac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon} \approx f'(x) + \dfrac{f^{(3)}(x)}{6}\varepsilon^2 + \dots =  f'(x) + O(\varepsilon^2) ~~(4)$

From this, if the derivative is approximated using formula $(3)$ (right
derivative approximation), the error will be $O(\varepsilon)$.
Meanwhile, if the derivative is approximated using formula $(4)$
(two-sided derivative approximation), the error will be
$O(\varepsilon^2) \ll O(\varepsilon)$ if $\varepsilon$ is small.

Both explanations above show that the two-sided derivative approximation
is a better approximation.

### For Multi-variable Functions {#sec:multi-variable}

For multi-variable functions, formula $(2)$ is applied to each variable
while keeping the other variables fixed. This method usually provides
quite accurate values. However, this method is not used to calculate
derivatives due to its high complexity compared to direct calculation.
When comparing this derivative with the exact derivative calculated
using the formula, people often reduce the dimensionality of the data
and the number of data points to facilitate calculations. Once the
calculated derivative is very close to the *numerical gradient*, we can
be confident that the calculated derivative is accurate.

Below is a simple code snippet to check the derivative, which can be
applied to any function (of a vector) with the `cost` and `grad`
calculated above.

```python
def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps 
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g 

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 

print('Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))
```

```python
Checking gradient... True
```

For other functions, readers only need to rewrite the `grad` and `cost`
functions above and apply this code snippet to check the derivative. If
the function is a function of a matrix, we need to make a slight change
in the `numerical_grad` function, which I hope is not too complicated.

For the Linear Regression problem, the derivative calculation as in
$(1)$ above is considered correct because the error between the two
calculations is very small (less than $(10^{-6}$. After obtaining the
correct derivative, we write the GD function:

```python
def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it) 

w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))
```

```python
Solution found by GD: w =  [[ 4.01780793  2.97133693]] ,
after 49 iterations.
```

After 49 iterations, the algorithm converges with a solution quite close
to the solution found using the formula.

Below is an animation illustrating the GD algorithm.

In the left image, the red lines are the solutions found after each
iteration.

In the right image, I introduce a new term: *level sets*.

### Level Sets {#sec:level-sets}

For the graph of a function with two input variables to be drawn in
three-dimensional space, it is often difficult to see the approximate
coordinates of the solution. In optimization, people often use a drawing
method that uses the concept of *level sets*.

If you pay attention to natural maps, to describe the height of mountain
ranges, people use many closed curves surrounding each other as follows:

The smaller red circles represent points at higher altitudes.

In optimization, this method is also used to represent surfaces in
two-dimensional space.

Returning to the GD algorithm illustration for the Linear Regression
problem above, the right image represents the level sets. That is, at
points on the same circle, the loss function has the same value. In this
example, I display the function's value at several circles. The green
circles have low values, and the outer red circles have higher values.
This is slightly different from natural level sets, where the inner
circles usually represent a valley rather than a mountain peak (because
we are looking for the smallest value).

I try with a smaller *learning rate*, and the result is as follows:

The convergence speed has slowed significantly, and even after 99
iterations, GD has not yet reached the best solution. In real-world
problems, we need many more iterations than 99 because the number of
dimensions and data points is usually very large.

**Another Example**

To conclude part 1 of Gradient Descent, I present another example.

The function $f(x, y) = (x^2 + y - 7)^2 + (x - y + 1)^2$ has two green
local minima at $(2, 3)$ and $(-3, -2)$, which are also two global
minima. In this example, depending on the initial point, we obtain
different final solutions.

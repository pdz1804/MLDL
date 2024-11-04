---
layout: post
title: 15. Sigmoid Function
mathjax: true
tags:
- Basic Machine Learning
categories: Basic_Machine_Learning
description: Details information about the Sigmoid Function
---

## Sigmoid functions

Sigmoid functions are the functions in mathematic that represent their
shape as letter \"S\". Most important advantage of applying sigmoid
functions in machine learning is that it is easy to get derivative which
lead to reduce to time complexity when computing their derivative in
learning process. We will go through three commonest sigmoid functions
including logistic function, hyperbolic tangent, and arctangent.

### Logistic Function

Logistic Function is a commonly used activation function in machine
learning, especially in classification problems. Its characteristic
S-shape makes it a smooth, continuous function that maps any real-valued
number to a value between 0 and 1, making it ideal for representing
probabilities.

The logistic function $\sigma(x)$ is defined as:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

where $e$ is the base of the natural logarithm, and $x$ is any
real-valued input.

**Key Characteristics**

**1. Range**: The output of the logistic function is always between 0
and 1: As $x \to +\infty$, $\sigma(x) \to 1$. As $x \to -\infty$,
$\sigma(x) \to 0$. At $x = 0$, $\sigma(x) = 0.5$.

**2. S-Shaped Curve**: The logistic function is often referred to as a
\"sigmoid\" because of its S-shaped curve. This makes it particularly
useful for probability estimation, as it maps large positive numbers
close to 1, large negative numbers close to 0, and values near zero
close to 0.5.

**3. Derivative**: The derivative of the sigmoid function has a specific
form, which makes it efficient to compute in backpropagation for neural
networks: $$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$ This form is
useful for calculating gradients, especially in binary classification
models, as it allows for faster computation and more efficient model
training.

**Applications**

**1. Logistic Regression**: The logistic sigmoid function is
foundational in logistic regression, where it is used to transform
linear combinations of input features into probabilities. This makes it
well-suited for binary classification tasks, where the output represents
the probability of belonging to a particular class.

**2. Neural Networks**: In early neural networks, sigmoid functions were
often used as activation functions in hidden layers. The output range of
(0, 1) allows for gradient-based learning and enables the network to
capture non-linear relationships.

**3. Probabilistic Interpretation**: Because the sigmoid output is
always between 0 and 1, it can be interpreted as the probability of an
instance belonging to a particular class. This probabilistic
interpretation is valuable in many applications, including spam
detection, medical diagnosis, and sentiment analysis.

**Advantages and Limitations**

\- **Advantages**: The logistic function is differentiable, allowing for
gradient-based optimization methods. In addition, its output range from
0 to 1 makes it ideal for probabilistic interpretations in
classification models.

\- **Limitations**: For very large or small input values, the logistic
function saturates, meaning the gradient approaches zero. This can slow
down or stop training in deep neural networks, particularly when many
layers are stacked. Unlike functions such as tanh, which are centered
around zero, the logistic sigmoid has an output range of (0, 1), which
can lead to slower convergence in some neural networks.

### The Hyperbolic Tangent (tanh) Function

The Hyperbolic Tangent (tanh) Function is widely used in machine
learning, especially in neural networks, due to its sigmoid shape and
symmetry around the origin. It is defined mathematically as:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

An alternative expression for the tanh function, in terms of the
logistic sigmoid function $\sigma(x)$, is:

$$\tanh(x) = 2 \cdot \sigma(2x) - 1$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid
function.

**Key Characteristics**

**1. Range**: The tanh function maps any real input to the range
$(-1, 1)$. Specifically: - For large positive inputs,
$\tanh(x) \approx 1$ - For large negative inputs,
$\tanh(x) \approx -1$ - For values near zero, $\tanh(x) \approx x$,
making it approximately linear around the origin.

**2. Symmetry**: The tanh function is an \*odd function\*, meaning it is
symmetric about the origin: $$\tanh(-x) = -\tanh(x)$$ This symmetry is
advantageous in cases where balanced outputs (centered around zero) are
preferred.

**3. Derivative**: The derivative of the tanh function is given by:
$$\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$$ This derivative has a
bell-shaped curve, with the largest slope at $x = 0$ and diminishing as
$x$ moves toward positive or negative infinity.

**Applications**

**1. Neural Networks**: The tanh function is often used as an activation
function in hidden layers. Since it maps inputs to a range between $-1$
and $1$, it allows neural networks to learn balanced patterns. This
property can improve the efficiency of gradient-based optimization
methods by centering the data.

**2. Signal Processing and Control Systems**: Tanh is suitable for
modeling processes that require both positive and negative output
ranges, making it applicable in signal processing and control systems.

**3. Image Processing**: In certain image processing tasks, tanh can
help normalize pixel values or be used in transformations where a
balanced range of pixel intensity is needed.

**Advantages and Limitations**

\- **Advantages**: The centered output of tanh allows for balanced data,
which can speed up convergence in gradient-based optimization. The range
of $(-1, 1)$ provides a symmetric output, allowing for positive and
negative signal propagation in neural networks.

\- **Limitations**: For very large or small input values, tanh outputs
values close to $\pm 1$, leading to near-zero gradients. This is known
as the vanishing gradient problem and can slow down or hinder training
in deep neural networks.

### Arctangent Function

Arctangent function (often denoted as $\arctan(x)$ or $\tan^{-1}(x)$) is
the inverse function of the tangent function. It is commonly used in
trigonometry and calculus and finds applications in fields such as
machine learning, image processing, and signal processing, where a
smooth, bounded sigmoid-like curve is desirable.

The arctangent function is defined as:

$$\arctan(x) = y \quad \text{such that} \quad \tan(y) = x \quad \text{and} \quad y \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$$

where $x$ is any real number, and $y$ is constrained to the range
$-\frac{\pi}{2}$ to $\frac{\pi}{2}$, ensuring that $\arctan(x)$ is
single-valued.

**Key Characteristics**

**1. Range**: The arctangent function maps real numbers to a bounded
range: $$y \in \left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$$ As
$x \to +\infty$, $\arctan(x) \to \frac{\pi}{2}$. As $x \to -\infty$,
$\arctan(x) \to -\frac{\pi}{2}$

**2. S-Shaped Curve**: Similar to other sigmoid functions like the
logistic and hyperbolic tangent functions, the arctangent has an
S-shaped curve. It is smooth, continuous, and symmetric around the
origin. This S-shape, coupled with the bounded range, makes it useful
for applications where controlled, gradual changes in output are needed.

**3. Derivative**: The derivative of the arctangent function is:
$$\frac{d}{dx} \arctan(x) = \frac{1}{1 + x^2}$$ This derivative
approaches zero as $x \to \pm \infty$, similar to other sigmoid-like
functions, giving it a bell-shaped curve.

### Applications 

**1. Machine Learning and Activation Functions**: Although less common
than logistic and tanh functions, the arctangent function can serve as
an activation function in neural networks, particularly for specialized
tasks requiring smooth, bounded output. It is valued for its smoother
slope near zero and its inherent symmetry.

**2. Signal Processing**: In signal processing, arctangent is used to
calculate phase angles, especially in applications involving complex
numbers. For example, the two-argument arctangent function,
$\text{atan2}(y, x)$, provides the phase angle of a complex number
$(x + iy)$ or the angle between a vector and the positive x-axis.

**3. Geometry and Robotics**: The arctangent function plays a role in
geometry and robotics for calculating angles, such as determining the
orientation of a robot based on its coordinates or finding angles
between points in space.

**4. Image Processing**: Arctangent is used in edge detection and image
gradient calculations to find the direction of intensity gradients in
images. The angle calculated via $\arctan$ provides orientation
information, which is useful for identifying edges and shapes within an
image.

### Advantages and Limitations

\- **Advantages**: With a range between $-\frac{\pi}{2}$ and
$\frac{\pi}{2}$, the arctangent function is well-suited for applications
requiring a controlled range. The output is symmetric about the origin,
which can be beneficial for applications requiring both positive and
negative outputs. Its continuous, smooth curve and easy-to-compute
derivative make it useful for applications needing gradual, non-linear
transformations.

\- **Limitations**: In some applications, the limited output range of
$-\frac{\pi}{2}$ to $\frac{\pi}{2}$ may not be ideal, especially if the
application requires a broader range. The arctangent function is less
commonly used as an activation function in deep neural networks, as
other functions like ReLU, tanh, and sigmoid are more widely optimized
for these tasks.

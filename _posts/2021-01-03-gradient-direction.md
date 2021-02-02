---
layout: post
title:  "Why the Gradient is the direction of steepest ascent?"
subtitle:   ""
categories: mathematics
tags: etc
---
1. Background
    - $$f(x,y) \in \mathbb{R}^2,\nabla f(x,y) = \begin{bmatrix}
    \frac{\partial f}{\partial x}\\ 
    \frac{\partial f}{\partial y}
    \end{bmatrix}$$에 대해, 어떤 vector $$\vec{v} = \begin{bmatrix}a&b \end{bmatrix}$$ 관점에서 편미분을 구하면, $$\nabla_{\vec{v}}f = \nabla f \cdot \vec{v}$$ 와 같다.
2. Description
    - $$f(x,y) \in \mathbb{R}^2,\nabla f(x,y) = \begin{bmatrix} \frac{\partial f}{\partial x}\\  \frac{\partial f}{\partial y} \end{bmatrix}$$
    에 대해, 어떤 unit vector $$\vec{v} = \begin{bmatrix} a& b \end{bmatrix}(\left \| \vec{v} \right \| = 1)$$
    가 있을 때, 벡터 $$\vec{v}$$에 의해 가장 function $$f$$의 값이 많이 변화하기 위해선 $$\nabla_{\vec{v}}f$$ 의 값이 최대가 되면 된다. $$\nabla_{\vec{v}}f = \nabla f \cdot \vec{v} = \left \| \nabla f \right \| \left \| \vec{v} \right \| cos\theta$$
    이므로, $$\vec{v}$$은 $$f$$과 같은 방향이 되게 된다.

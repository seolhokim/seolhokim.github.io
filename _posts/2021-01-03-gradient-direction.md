---
layout: post
title:  "Why the Gradient is the direction of steepest ascent?"
description: "gradient가 steepest ascent 방향인 이유를 설명한 글. 방향 도함수를 gradient와 단위 벡터 v의 내적으로 나타내고, 그 값이 cosθ에 비례하므로 v가 gradient와 같은 방향일 때 함수 값의 변화가 가장 커짐을 짧은 수식으로 보인다."
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author: Seolho Kim
math: true
---
## Background
    - $$f(x,y) \in \mathbb{R}^2,\nabla f(x,y) = \begin{bmatrix}
    \frac{\partial f}{\partial x}\\ 
    \frac{\partial f}{\partial y}
    \end{bmatrix}$$에 대해, 어떤 vector $$\vec{v} = \begin{bmatrix}a&b \end{bmatrix}$$ 관점에서 편미분을 구하면, $$\nabla_{\vec{v}}f = \nabla f \cdot \vec{v}$$ 와 같다.
## Description
    - $$f(x,y) \in \mathbb{R}^2,\nabla f(x,y) = \begin{bmatrix} \frac{\partial f}{\partial x}\\  \frac{\partial f}{\partial y} \end{bmatrix}$$
    에 대해, 어떤 unit vector $$\vec{v} = \begin{bmatrix} a& b \end{bmatrix}(\left \| \vec{v} \right \| = 1)$$
    가 있을 때, 벡터 $$\vec{v}$$에 의해 가장 function $$f$$의 값이 많이 변화하기 위해선 $$\nabla_{\vec{v}}f$$ 의 값이 최대가 되면 된다. $$\nabla_{\vec{v}}f = \nabla f \cdot \vec{v} = \left \| \nabla f \right \| \left \| \vec{v} \right \| cos\theta$$
    이므로, $$\vec{v}$$은 $$f$$과 같은 방향이 되게 된다.

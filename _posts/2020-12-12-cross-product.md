---
layout: post
title:  "10. Cross product"
subtitle:   ""
categories: mathematics
tags: linearalgebra
---
## Cross product

1. **What to learn**
    - What is the Cross product
2. **Description**
    - Notation
        - $$\vec{x} \times \vec{y} = \vec{z}$$
    - New vector that length is same the parallelagram that two vectors make. It's direction is decided by right hand rule.
    - How to calculate
        - Determinant is needed.
            - About two vector $$ \vec{x} = \begin{bmatrix}
            x_1 & x_2
            \end{bmatrix}$$, $$\vec{y} = \begin{bmatrix}
            y_1 & y_2
            \end{bmatrix}$$, we can think it as transformation $$\begin{bmatrix}
            x_1 & y_1 \\ x_2 & y_2
            \end{bmatrix}$$. And find how square area that basis vector make is changed.
    - $$\vec{z}$$ is bigger when two vector $$\vec{x}$$ and $$\vec{y}$$ is perpendicular.
    - $$\vec{z}$$ is linear about length of two vector $$\vec{x}$$ and $$\vec{y}$$.
    - To make it more general, $$\vec{x} = \begin{bmatrix}
    x_1 & x_2 & x_3
    \end{bmatrix}$$ and $$\vec{y} = \begin{bmatrix}
    y_1 & y_2 & y_3
    \end{bmatrix}$$. So,
    - $$\vec{x}\times\vec{y} = \vec{z}$$ is

        $$\begin{bmatrix}
        x_1 \\ x_2 \\ x_3
        \end{bmatrix} \times \begin{bmatrix}
        y_1 \\ y_2 \\ y_3
        \end{bmatrix} = \begin{bmatrix}
        x_2y_3 - x_3y_2 \\ x_3y_1-x_1y_3 \\ x_1y_2-x_2y_1
        \end{bmatrix}$$

        next step, we will talk about why this calculation appears.

3. **Next Step**
    - Cross products in the light of linear transformations
4. **References**
    - [https://www.youtube.com/watch?v=eu6i7WJeinw&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=10](https://www.youtube.com/watch?v=eu6i7WJeinw&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=10)
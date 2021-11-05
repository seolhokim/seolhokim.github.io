---
layout: post
title:  "11. Cross products in the light of linear transformations"
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author:
  - Seolho Kim
math: true
---
## Cross products in the light of linear transformations

1. **What to learn**
    - How can we derive **product dot**?
2. **Description**
    - Before we saw this.

    $$\begin{bmatrix}
    v_1 \\ v_2 \\ v_3
    \end{bmatrix} \times \begin{bmatrix}
    w_1 \\ w_2 \\ w_3
    \end{bmatrix} = det\left (  \begin{bmatrix}
    \vec{i} & v_1 & w_1 \\ \vec{j} & v_2 & w_2 \\\vec{k} & v_3 & w_3 
    \end{bmatrix}\right )$$

    $$\vec{i}(v_2w_3-v_3w_2) + \vec{j}(v_3w_1-v_1w_3) + \vec{k}(v_1w_2-v_2w_1)$$

    - Let's think about function $$f(\vec{x})$$, derminant of $$\vec{x},\vec{v},\vec{w}$$.

        $$f\left (  \begin{bmatrix}
        x  \\ y  \\ z 
        \end{bmatrix}\right ) = det\left (  \begin{bmatrix}
        x & v_1 & w_1 \\ y & v_2 & w_2 \\z & v_3 & w_3 
        \end{bmatrix}\right ) 
        \cdots (1)$$

        We can get volumn of parallelepiped using function $$f(\vec{x})$$. Area of parallelepiped is calculated by (area of parallelogram) X (component of $$\begin{bmatrix}
        x  & y  &z 
        \end{bmatrix}^T$$ perpendicular to $$\vec{v}$$ and $$\vec{w}$$). → $$f(\vec{x})$$ is calculated by $$\vec{p} = \begin{bmatrix}
        v_1  \\ v_2\\ v_3 
        \end{bmatrix} \times \begin{bmatrix}
        w_1  \\ w_2  \\ w_3 
        \end{bmatrix}$$ and, do inner dot with $$\begin{bmatrix}
        x  \\ y \\ z
        \end{bmatrix}$$. Then we can express this like this.

        $$f\left (  \begin{bmatrix}
        x  \\ y  \\ z 
        \end{bmatrix}\right )=\begin{bmatrix}
        p_1  & p_2&p_3
        \end{bmatrix} \cdot \begin{bmatrix}
        x  \\ y\\ z
        \end{bmatrix} = p_1x+p_2y+p_3z \cdots(2)$$

        Looking at (1), we can get $$x(v_2w_3-v_3w_2) + y(v_3w_1-v_1w_3) + z(v_1w_2-v_2w_1) \cdots (3)$$.

        Looking at (2), we find $$\vec{p}$$,    $$\begin{matrix}
        p_1 = v_2 \cdot w_3 - v_3 \cdot w_2 \\ 
        p_2 = v_3\cdot w_1 - v_1 \cdot w_3\\ 
        p_3 = v_1 \cdot w_2 - v_2 \cdot w_1
        \end{matrix}$$

        So, we find $$\vec{p}$$.

3. **Next Step**
4. **References**
    - [https://www.youtube.com/watch?v=BaM7OCEm3G0&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=11](https://www.youtube.com/watch?v=BaM7OCEm3G0&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=11)

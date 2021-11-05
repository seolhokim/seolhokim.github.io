---
layout: post
title:  "8. Nonsquare matrices as transformations between dimensions"
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author:
  - Seolho Kim
math: true
---
## Nonsquare matrices as transformations between dimensions

1. **What to learn**
    - What about nonsquare matrices?
2. **Description**
    - example

        $$\begin{bmatrix}2 & 0\\  -1&1 \\  -2& 1\end{bmatrix}$$

        - The column space of this matrix place where all the vectors land is a 2d plane slicing through the origin of 3d space
        - But this matrix is still full rank, because the column space is the same as the number of dimensions of the input space.
        - So geometric interpretation of mapping two dimensions to three dimensions since the two columns indicate that the input space has two basis vectors and the three rows indicate that the landing spots for each of those basis vectors is described with three separated coordinates.

            $$\begin{bmatrix}3 & 1&4\\  1&5&9\end{bmatrix}$$

        - Three columns indicate 3 basis vectors  that we're starting dimension.
        - Two rows indicates that the landing spot for each of those three basis vectors.
        - So it must all vectors will be landing in two dimensions
    - We could also have a transformation from two dimensions to one dimension
        - example

            $$\begin{bmatrix}1 &2\end{bmatrix}$$

            - It will land i hat on 1, j hat on 2. And it represent same direction. so we can add together.
3. **Next Step**
    - Dot products and duality
4. **References**
    - [https://www.youtube.com/watch?v=v8VSDg_WQlA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8](https://www.youtube.com/watch?v=v8VSDg_WQlA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8)
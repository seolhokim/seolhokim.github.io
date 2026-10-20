---
layout: post
title:  "5. Three-dimensional linear transformations"
description: "3Blue1Brown 선형대수 시리즈 5편 정리. i hat, j hat, k hat 세 basis vector가 옮겨지는 위치로 3차원 linear transformation을 matrix로 나타내고, 3x3 matrix 곱 결과를 열별로 유도하는 과정을 정리한 노트."
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author: Seolho Kim
math: true
---
## Three-dimensional linear transformations

## What to learn 
    - How to do Three dimensional Matrix multiplication
## Description
    - We already know multipling 2d matrix.
    - Three-dimensional linear transformation applies the same principle.
    - basis vector
        - i hat
        - j hat
        - k hat

        $$\begin{bmatrix}a & b & c\\ d & e & f\\ g & h & i\end{bmatrix}\begin{bmatrix}
        j & k & l\\ 
        m & n & o\\ 
        p & q & r
        \end{bmatrix}$$

        - [j m p]^T is that transformed i hat laned vector. and we want to know [j m p]^T will go.

        $$j\begin{bmatrix}
        a \\  d\\ g
        \end{bmatrix} + m\begin{bmatrix}
        b \\  e\\ h
        \end{bmatrix} + p\begin{bmatrix}
        c \\  f\\ i
        \end{bmatrix}$$

        - we can do iteratively to j hat and k hat, and get matrix multiplication result.

        $$\begin{bmatrix}aj+bm+cp & ak+bn+cq & al+bo+cr\\ dj+em+fp & dk+en+fq & dl+eo+fr\\ gj+hm+ip & gk+hn+iq & gl+ho+ir\end{bmatrix}$$

## Next Step
- The determinant
## References
- [https://www.youtube.com/watch?v=rHLEWRxRGiM&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=5](https://www.youtube.com/watch?v=rHLEWRxRGiM&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=5)
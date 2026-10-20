---
layout: post
title:  "4. Matrix multiplication as composition"
description: "3Blue1Brown 선형대수 시리즈 4편 정리. 두 matrix의 곱을 f(g(x))처럼 오른쪽부터 적용하는 변환의 합성으로 해석하고, basis vector가 옮겨지는 위치로 2x2 matrix 곱 공식을 유도하는 과정을 정리한 선형대수 노트."
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author: Seolho Kim
math: true
---
## Matrix multiplication as composition

## What to learn
- How to do Matrix multiplication

## Description
- We already know multipling matrix by vector.
- Multiple transformation using matrixs can be interpreted by Only one matrix.
- Think about composite function like f(g(x)). we read this from right to left. matrix multiplication has same property. Make imaginary basis(i hat and j hat) and multipling matrix by vector from right to left.

  $$\begin{bmatrix}a & b\\ c & d\end{bmatrix}\begin{bmatrix}e & f\\ g & h\end{bmatrix} = \begin{bmatrix}a & b\\ c & d\end{bmatrix}\begin{bmatrix}e\\ g\end{bmatrix} , \begin{bmatrix}a & b\\ c & d\end{bmatrix}\begin{bmatrix}f\\ h\end{bmatrix}$$

  $$\begin{bmatrix}a & b\\ c & d\end{bmatrix}\begin{bmatrix}e\\ g\end{bmatrix} , \begin{bmatrix}a & b\\ c & d\end{bmatrix}\begin{bmatrix}f\\ h\end{bmatrix} = e \begin{bmatrix}a\\ b\end{bmatrix} + g \begin{bmatrix}b\\ d\end{bmatrix} , f \begin{bmatrix}a\\ b\end{bmatrix} + h \begin{bmatrix}b\\ d\end{bmatrix} $$

  $$e \begin{bmatrix}a\\ c\end{bmatrix} + g \begin{bmatrix}b\\ d\end{bmatrix} , f \begin{bmatrix}a\\ c\end{bmatrix} + h \begin{bmatrix}b\\ d\end{bmatrix} = \begin{bmatrix}
  ae+bg & af+bh\\ 
  ce+dg & cf+dh
  \end{bmatrix}$$

## Next Step
- Three-dimensional linear transformations
## References
- [https://www.youtube.com/watch?v=XkY2DOUCWMU&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=4](https://www.youtube.com/watch?v=XkY2DOUCWMU&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=4)

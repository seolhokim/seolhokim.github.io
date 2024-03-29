---
layout: post
title:  "4. Matrix multiplication as composition"
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author:
  - Seolho Kim
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

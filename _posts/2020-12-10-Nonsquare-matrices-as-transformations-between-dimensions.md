---
layout: post
title:  "8. Nonsquare matrices as transformations between dimensions"
description: "3Blue1Brown 선형대수 시리즈 8편 정리. 3x2, 2x3, 1x2 matrix 예시로 nonsquare matrix를 차원 사이의 변환으로 해석하고, column과 row 수가 입력 차원과 출력 좌표 수를 뜻한다는 점, full rank 판단을 정리한 노트."
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author: Seolho Kim
math: true
---
## Nonsquare matrices as transformations between dimensions

## What to learn
- What about nonsquare matrices?
## Description
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
## Next Step
- Dot products and duality
## References
- [https://www.youtube.com/watch?v=v8VSDg_WQlA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8](https://www.youtube.com/watch?v=v8VSDg_WQlA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8)
---
layout: post
title:  "13. Change of basis"
description: "3Blue1Brown 선형대수 시리즈 13강 Change of basis 정리. 행렬 A로 변환된 좌표계의 벡터를 원래 관점으로 옮기거나 역행렬로 되돌리는 예시와, 다른 좌표계에서 본 변환을 A^-1BA 형태로 표현하는 방법을 설명한 선형대수 개념 정리 글."
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author: Seolho Kim
math: true
---
## Change of basis

## What to learn
- Perspective of basis and vector

## Description
- If we want to see vector in transformed coordinates by $$A$$ by transformed perspective, we have to calculate the inverse of $$A$$.
  - (example)
    - In Transformed coordinates by $$A = \begin{bmatrix}
      2  & -1 \\ 1 & 1
      \end{bmatrix}$$, we can find vector $$\vec{v} = \begin{bmatrix}
      -1  & 2
      \end{bmatrix}^T$$ in transformed perspective. Vector $$\vec{v}$$ that is made by transformed $$-\vec{i} + 2\vec{j}$$.  So, doing $$A\cdot \vec{v}$$  , we can get $$\vec{w} = A\cdot \vec{v} = \begin{bmatrix}
      -4  & 1
      \end{bmatrix}$$ in normal perspective.
      - In transformed coordinates by $$A = \begin{bmatrix}
      2  & -1 \\ 1 & 1
      \end{bmatrix}$$, if someone see our vector $$\vec{v} = \begin{bmatrix}
      3  & 2
      \end{bmatrix}^T$$, then $$A^{-1}v$$ vector is needed.
- How can we do another transformation in transformed coordinates by $$A = \begin{bmatrix}
  a  & b \\ c & d
  \end{bmatrix}$$?
  - Get transformed vector in normal perspective.

    $$\begin{bmatrix}
    a  & b \\ c & d
    \end{bmatrix} \begin{bmatrix}
    x  \\ y
    \end{bmatrix}$$
    
  - apply transformation $$B = \begin{bmatrix}
    e  & f \\ g & h
    \end{bmatrix}$$

    $$\begin{bmatrix}
    e  & f \\ g & h
    \end{bmatrix}\begin{bmatrix}
    a  & b \\ c & d
    \end{bmatrix} \begin{bmatrix}
    x  \\ y
    \end{bmatrix}$$

  - Inner dot with inverse of A makes it to transformed coordinates.

    $$\begin{bmatrix}
    a  & b \\ c & d
    \end{bmatrix}^{-1}\begin{bmatrix}
    e  & f \\ g & h
    \end{bmatrix}\begin{bmatrix}
    a  & b \\ c & d
    \end{bmatrix} \begin{bmatrix}
    x  \\ y
    \end{bmatrix}$$

## Next Step
- Eigenvectors and eigenvalues
## References
- [https://www.youtube.com/watch?v=P2LTAUO1TdA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=13](https://www.youtube.com/watch?v=P2LTAUO1TdA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=13)
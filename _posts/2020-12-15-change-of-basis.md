---
layout: post
title:  "13. Change of basis"
subtitle:   ""
categories: mathematics
tags: linearalgebra
---
## Change of basis

1. **What to learn**
    - Perspective of basis and vector
2. **Description**
    - If we want to see vector in transformed coordinates by $$A$$ by transformed perspective, we have to calculate the inverse of $$A$$.

        (example)

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

3. **Next Step**
    - Eigenvectors and eigenvalues
4. **References**
    - [https://www.youtube.com/watch?v=P2LTAUO1TdA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=13](https://www.youtube.com/watch?v=P2LTAUO1TdA&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=13)
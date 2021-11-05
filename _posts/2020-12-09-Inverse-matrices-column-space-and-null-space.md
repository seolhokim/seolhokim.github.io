---
layout: post
title:  "7. Inverse matrices, column space and null space"
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author:
  - Seolho Kim
math: true
---
## Inverse matrices, column space and null space

## What to learn
- What is Inverse matrices
- What is Column Space
- What is Rank
- What is null space
## Description
- Why linear algebra is useful?
  - It lets us solve certain systems of equations
- System of equations
  - we have list of variables and equations relating them
  - Linear System of equations example

    $$\begin{matrix}
            2x+5y+3z = -3\\ 
            4x+0y+8z = 0\\ 
            1x+3y+0z=2
    \end{matrix}$$

    $$\rightarrow \begin{bmatrix}2 & 5 & 3\\ 4 & 0 & 8\\ 1 & 3 & 0\end{bmatrix}\begin{bmatrix}
            x\\ 
            y\\ 
            z
            \end{bmatrix} = \begin{bmatrix}
            3\\ 
            0\\ 
            1
    \end{bmatrix}$$

    $$\rightarrow A\vec{x} = \vec{v}$$

  - It means find vector x using transformed vector v by linear transformation A
  - How we find solution of this equation is depands on whether the transformation A squish space to lower dimension or not.(first step we have to find) → case determinant is zero or not
    - Nonzero case
      - Linear transformation is one-to-one correspondence
        - Only one vector is corresponded to Only one transformated vector.
      - Only one solution in $$x = A^{-1} v$$ always exists. and we can call $$A^{-1}$$ as the **inverse of A**
      - $$A^{-1} * A$$ equals the transformation that does nothing. → identity transformation
      - $$det(A) ≠ 0$$, A doesn't squish the space
    - **Zero case**
      - In $$det(A) = 0$$ case, A squish the space to lower dimension. So we can't find Inverse of A. we can't unsquish line to plane using function.
        - Because single vector has to become multiple vector to broaden dimension.
      - But we can find the infinite solutions if transformation squish a plane to line while x and v are same direction(linearly dependant) and others do not exist solution.
    - **Rank**
      - Number of dimensions in the output
        - we want to know after transformation, how many dimensions are squished.
        - When the output of transformation is line, means one dimensional, then we can say transformation's rank is 1
        - if All the vectors land on two dimensional plane, then we can say transformation's rank is 2
      - Max rank of 2x2 transformation is 2
    - **Column space**
      - Set of all possible outputs Ax
      - **Span of columns  ↔ Column space**
        - **Rank** is number of dimensions in the **column space**
        - If column space equals number of column, then we can say it as **full rank**
    - **Zero vector**
      - It is always included in the column space
        - Because linear transformation always include fixed origin.
      - For a full rank transformation, only vector to origin itself is the zero vector.
    - **Null space = kernel**
      - If transformation makes vector to origin, then we can say that set of vectors as **Null space**
        - If plane is squished to line, then there are lots of vectors in same line become zero vector.
        - If space is squished to plane, then there are also lots of vectors in same line become zero vector.
        - If space is squished to line, then there are lots of vectors in plane vectors become zero vector.
      - $$A \vec{x} = \vec{0}$$ → all null space be possible solution.
  - **Overview**
    - Each system has some kind of linear transformation associated with it. and when that transformation has an inverse, we can use that inverse to solve our system.
    - column space lets us understand when a solution even exists.
    - null space helps us understand what the set of all possible solution can look like.
## Next Step
- Nonsquare matrices as transformations between dimensions
- Gaussian elimination and Row echelon form
## References
- [https://www.youtube.com/watch?v=rHLEWRxRGiM&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=5](https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=6)
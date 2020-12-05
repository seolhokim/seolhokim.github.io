---
layout: post
title:  "Linear transformations and matrices"
subtitle:   ""
categories: deeplearning
tags: remind
---
## Linear transformations and matrices
1. **What to learn**
    - What is the **Linear Transformation**
2. **Description**
    - **Transformation**
     - Take some input vector to output vector
      - = input vector moving over output vector
     - We will treat Only **Linear Transformation** In Linear Algebra
      - Special case of Transformation
       - Lines remain lines
       - Origin remains fixed
      - 2d Linear Transformation is fully discribed by two basis vectors
       - Always same linear combination about two basis vectors
       - And if we can explain 2d Linear Transformation using **Matrix,** and we can find out where a specific vector will move.
        - Example
         - If we make 2d linear transformation that move i hat to [3,2]^T and j hat to [2,1]^T. and wonder where [5,7]^T will move.
         - $$ \vec{i} = \begin{bmatrix}1\\ 0\end{bmatrix},\vec{j} = \begin{bmatrix}0\\ 1\end{bmatrix} \rightarrow \vec{i} = \begin{bmatrix}3\\ 2\end{bmatrix},\vec{j} = \begin{bmatrix}2\\ 1\end{bmatrix} $$
         - $$ 5 * \begin{bmatrix}3\\ 2\end{bmatrix} + 7 \begin{bmatrix}2\\ 1\end{bmatrix} = 
                        \begin{bmatrix}29\\ 17\end{bmatrix} $$
         - [29,17]^T will be the answer. Another way, we can make matrix using i hat and j hat
         - $$ \begin{bmatrix}3\\ 2\end{bmatrix}\begin{bmatrix}2\\ 1\end{bmatrix}\rightarrow \begin{bmatrix}3 & 2\\ 2 & 1\end{bmatrix} $$
         - Then we can get result using matrix calculation.
         - $$ \begin{bmatrix}3 & 2\\ 2 & 1\end{bmatrix}\begin{bmatrix}5\\ 7\end{bmatrix} = 
                        \begin{bmatrix}29\\ 17\end{bmatrix} $$
         - This way, we can get same result.
       - If matrix is compounded of Linearly dependent columns, then it makes all vectors move to the vectors' span.
3. **Next Step**

4. **References**
    - [https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3](https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3)

---
layout: post
title:  "12. Cramer's rule"
subtitle:   ""
categories: mathematics
tags: linearalgebra
---
## Cramer's rule, explained geometrically

1. **What to learn**
    - Understand Cramer's rule by geometrically
2. **Description**
    - Gaussian elimination is always fast. But understanding cramer's rule geometrically will help to stretch out our linear algebra skill.
    - About linear transformation $T$,

        $$\begin{bmatrix}
        x  \\ y  
        \end{bmatrix} \cdot \begin{bmatrix}
        1  \\ 0  
        \end{bmatrix} \neq T\left ( \begin{bmatrix}
        x  \\ y  
        \end{bmatrix} \right) \cdot T \left ( \begin{bmatrix}
        1  \\ 0  
        \end{bmatrix}\right)$$

        But, if $T$ is orthonormal(orthogonal and unit), then 

        $$\begin{bmatrix}
        x  \\ y  
        \end{bmatrix} \cdot \begin{bmatrix}
        1  \\ 0  
        \end{bmatrix} = T\left ( \begin{bmatrix}
        x  \\ y  
        \end{bmatrix} \right) \cdot T \left ( \begin{bmatrix}
        1  \\ 0  
        \end{bmatrix}\right)$$

        and, Any orthonormal linear transformation $A$, 

        $$\begin{bmatrix}
        a & b \\c&d  
        \end{bmatrix} \begin{bmatrix}
        x\\y
        \end{bmatrix} = \begin{bmatrix}
        e\\f
        \end{bmatrix}$$

        we can find misterious vector $\vec{v}$ by inner dot 

        $$x = \begin{bmatrix}
        e\\f
        \end{bmatrix}\begin{bmatrix}
        a\\b
        \end{bmatrix}, y = \begin{bmatrix}
        e\\f
        \end{bmatrix}\begin{bmatrix}
        b\\d
        \end{bmatrix}$$

        Because, basis of transformed $\vec{x}$ is same as the result of inner dot $\begin{bmatrix}
        e\\f
        \end{bmatrix}$ with new basis.

    - Yellow area is 1 * y

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ded98dce-14c2-47bc-83f2-41fc79620ff2/linear_algebra_7.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ded98dce-14c2-47bc-83f2-41fc79620ff2/linear_algebra_7.png)

        Then, after transformated, we can calculate area of  yellow area transformed by $A$ is $det(A)y$.

        - Because, it is linear transformation. $1 \times y$ stretch out $det(A) \times 1 \times y$

        So, $y = \frac{Area}{det(A)}$, Area is (landing vector $\vec{v}$) $\cdot$(transformed i hat). We can describe like this.

        $$y = \frac{Area}{det(A)} = \frac{\left( \begin{bmatrix}
        a & v_1 \\c&v_2  
        \end{bmatrix}\right )}{\left ( \begin{bmatrix}
        a & b \\c&d  
        \end{bmatrix}\right )}$$

        And we can get x same as y.

        $$x = \frac{Area}{det(A)} = \frac{\left( \begin{bmatrix}
        v_1 & b \\v_2&d  
        \end{bmatrix}\right )}{\left ( \begin{bmatrix}
        a & b \\c&d  
        \end{bmatrix}\right )}$$

3. **Next Step**
    - Think about 3d Cramer's rule
    - Change of basis
4. **References**
    - [https://www.youtube.com/watch?v=jBsC34PxzoM&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=12](https://www.youtube.com/watch?v=jBsC34PxzoM&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=12)
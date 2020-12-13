---
layout: post
title:  "14. Eigenvectors and eigenvalues"
subtitle:   ""
categories: mathematics
tags: linearalgebra
---
## Eigenvectors and eigenvalues

1. **What to learn**
    - What is the eigenvectors
    - What is the eigenvalues
    - What is the eigenbasis
2. **Description**
    - After transformation by matrix $$A$$, some vectors remain in their own span. thoses are called by eigenvectors. And they have their own eigenvalues the factor is how much they are stretched or squished during transformation.
    - Think about rotation in three dimension. then eigenvector is the axis of rotation. and the eigenvalue is one.
    - As described above, it is expressed in a formular like this. $$A\vec{v} = \lambda I\vec{v}$$ , $$\lambda$$ is eigenvalue, and $$\vec{v}$$ is eigenvector, and $$A$$ is transformation.
    - $$(A-\lambda I)\vec{v} = \vec{0}$$
    - We want a nonzero solution for $$\vec{v}$$, so matrix $$A-\lambda I$$ has to squish the space into lower dimension to make zero vector by nonzero vector.
    - → $$det(A-\lambda I) = 0$$

        $$det\left(\begin{bmatrix}
        a -\lambda & b \\ c & d-\lambda
        \end{bmatrix}\right) = (a-\lambda)(d-\lambda) -bc = 0$$

        if any $$\lambda$$ makes above fomular, then the eigenvector $$\vec{v}$$ will be stratched or squished by $$\lambda$$. And if we plug in the $$\lambda$$ into matrix, then the eigenvector $$\vec{v}$$ will be kernel.

    - $$\lambda$$ may exist only one((ex) sheer matrix) or not((ex)rotation matrix).
    - Eigenvector can be multiple or none. if $$2I$$ stretch all vector in coordinates. so eigenvector is all vectors.
    - Eigenbasis
        - A set of basis vectors, which are also eigenvertors.
        - Diagonal matrix makes all basis vectors to eigenvectors. and diagonal entries are eigenvalues.
            - Diagonal matrix has powerful computation property.
                - Diagonal matrix $$A = \begin{bmatrix}
                a& 0 \\ 0 & b
                \end{bmatrix}$$, $$A^{100} = \begin{bmatrix}
                a^{100}& 0 \\ 0 & b^{100}
                \end{bmatrix}$$
            - To apply it, when we calculate power of $$B$$, we transform the basis,easily find the result.

                $$A^{-1}BA = C, C^{100} = A^{-1}B^{100}A$$

                matrix $$A$$ changes basis, and transforms by $$B$$, and inverse by $$A^{-1}$$.

3. **Next Step**
    - Abstract vector spaces
4. **References**
    - [https://www.youtube.com/watch?v=PFDu9oVAE-g&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=14](https://www.youtube.com/watch?v=PFDu9oVAE-g&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=14)
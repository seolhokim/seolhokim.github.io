---
layout: post
title:  "6. The determinant"
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author:
  - Seolho Kim
math: true
---
## The determinant

## What to learn
- what is the **determinant**?
## Description
    - 2d **determinant** make us estimate transformed area of region.
        - example
            1. 

                $$det\left ( \begin{bmatrix}0.0 & 2.0\\ -1.5 & 1.0\end{bmatrix} \right ) = 3.0$$

                - This transformation increases the area of A region by a factor of three.
            2. 

                $$det\left ( \begin{bmatrix}0.5 & 0.5\\ -0.5 & 0.5\end{bmatrix} \right ) = 0.5$$

                - This transformation squishes down all areas by a factor of 1/2
            3. 

                $$det\left ( \begin{bmatrix}4 & 2\\ 2 & 1\end{bmatrix} \right ) = 0$$

                - This transformation squishes all of space onto a line or onto a single point.
    - **Determinant** can get negative number.
        - Negative determinant invert the orientation space
        - Originally j hat is left side of i hat. but inverted by transformation that has nagative determinant, then j hat is right side of i hat. → Orientation space is inverted.
    - If we draw graph that shows **determinant** changes from positive to negative, space will be squished, then will be line, then will be flipped.
    - Determinant formular is ad-bc. we can get this formular using vector area diagram.
- In three dimension, this idea is maintained.
    - **Determinant** informs the volume of basis vectors and if it is zero, then it means column vectors must be linear dependent.
    - We can define basis direction by right hand rule.
        - index finger : i hat
        - midde finger : j hat
        - thumb : k hat
    - If Determinant is smaller than zero, we have to change right hand rule to using left hand(Orientation flipped)

    $$det\left ( \begin{bmatrix}a & b &c\\ d & e &f \\ g&h&i\end{bmatrix} \right ) = a \cdot det\left ( \begin{bmatrix}e & f\\ h & i\end{bmatrix} \right ) - b\cdot det\left ( \begin{bmatrix}d & f\\ g & i\end{bmatrix} \right )  + c\cdot det\left ( \begin{bmatrix}d & e\\ g & h\end{bmatrix} \right )$$

## Next Step
- Inverse matrices, column space and null space
## References
- [https://www.youtube.com/watch?v=rHLEWRxRGiM&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=5](https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=6)

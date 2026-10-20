---
layout: post
title:  "15. Abstract vector spaces"
description: "3Blue1Brown 선형대수 시리즈 15강 Abstract vector spaces 정리. 다항식 같은 함수도 기저를 달리한 벡터로 볼 수 있다는 예시와, 덧셈과 스칼라 곱에 관한 8가지 벡터 공리를 만족하면 무엇이든 벡터로 다룰 수 있음을 설명한 글."
toc: true
categories: 
  - Mathematics
tags: [Linear Algebra]
author: Seolho Kim
math: true
---
## Abstract vector spaces

### What to learn
- Back to the beginning, what is vector?
### Description
- Function is vector?
  - Yes!
  - It is another type of vector with different basis.
    - example : In polynomial space, we can express

    $$x^2 + 3x + 5 = \begin{bmatrix}
    5\\3\\1
    \end{bmatrix},b_0(x) = 1,b_1(x)=x,b_2(x) = x^2$$

  - We can say all of things as vector that satisfy axiom of vector!
    - Rule for vectors addition and scaling
      - $$\vec{u} + (\vec{v}+\vec{w}) = (\vec{u} + \vec{v})+\vec{w}$$
      - $$\vec{v}+\vec{w} = \vec{w}+\vec{v}$$
      - $$\vec{0} + \vec{v} = \vec{v}$$
      - $$- \vec{v} + \vec{v} = \vec{0}$$
      - $$a(b\vec{v}) = (ab)\vec{v}$$
      - $$1\vec{v} = \vec{v}$$
      - $$a(\vec{v}+\vec{w} ) = a\vec{v}+a\vec{w}$$
      - $$(a+b)\vec{v} = a\vec{v} + b\vec{v}$$
    - In the modern theory the form that vectors take doesn't really matter.
## Next Step
- review! and solve a mount of problems!
## References
- [https://www.youtube.com/watch?v=PFDu9oVAE-g&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=15](https://www.youtube.com/watch?v=TgKwz5Ikpc8&t=902s)
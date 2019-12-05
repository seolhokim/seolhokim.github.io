---
layout: post
title:  "Image and Kernel of a Linear tansformation-1"
subtitle:   "image와 kernel 이해 하기"
categories: deeplearning
tags: lecture
---

선형대수를 학교에서 배웠었는데 기억이 하나도 안나서 다시 가끔가끔 내용정리를 해보려고한다.. 또한
눈으로봐도 대충 이해는 되는데 공역 치역등으로 배웠어서 영문 책을 보니
codomain이 처음엔 도대체 뭔가 싶었다. 어차피 영문으로 다시 배워야 하니 간단한 내용부터 정리해야겠다.



# Image의 정의

f : X-> Y 에대해,  $$ x \in  X$$ 일때, $$ f(x) \in Y $$ 를 x의 image 라고한다!

(어렵게 생각했었는데 사실 image란 치역이었다.. y = f(x)의 y를 x의 image라고 해석하면됨)

mapping 은 correspondence과 같은 개념으로, 대응시킨다고 이해하면 편하다. 대응 시키는 집합들의 관계에 따라
1대1 다대다 등이 있다.

domain과 codomain은 정의역과 공변이다 치역은 range로 image라고도 한다

transform은 정의역과 공변이 같은 mapping이다. kernel은 image로 mapping 될때

0로 mapping 되는 domain의 부분집합 L에 대해 kernel이라고 한다.(집합이다)

여기서 중요한 정리인

rank-nullity 정리가 나오는데

L : V -> U

Rank(L) + Nullity(L) = Dim(V) 이다

$$ A^-1 $$ 가 존재하기 위해선  $$ det(A) != 0 $$ 이어야한다.

determinant 에는 많은 재밌는 성질이 존재하는데 기하학적으로 transform될 때

꼭지점의 방향이라든지 선이 되버린다던지 하는 성질이 있다.

trace 는 대각합!

-------------------------------------------

모든 벡터공간은 그 안에 속하는 일정한 갯수의 벡터들의 일차결합으로 표현할 수 있고, 그 벡터들의 집합은
벡터 공간의 span이라고 한다.

이때, 벡터들이 일차 종속인가 일차 독립인가 구별할 수 있는데, 일차 독립이면 그 집합을 기저라고한다.


-------------------------------------------

eigenvalue, eigenvector

는 고유값과 고유벡터로, 행렬 A가 선형변환일 때, 선형변환 A에 대한 변환 결과가 자기 자신의 상수배가 되는 0이 아닌 벡터를 고유벡터라하고, 이 상수배를 고유값이라고 한다.

--------------------------------------------

svd (Singular Value Decomposition)

고유값 분해처럼 행렬 대각화 하는 방법으로 직사각형 행렬일 때도 가능하다

$$ A = U \sum{V^T} = u_1 \sigma_1 v_1^T + u_2 \sigma_2 v_2^T +...+ u_r \sigma_r v_r^T $$

A : m*n 직사각 행렬,

U : A의 left singular vector로 이루어진 m * m 직교행렬

$$\sum $$ : 주대각성분이 $$ 너무피곤해서 일단자러감 





[https://staff.csie.ncu.edu.tw/chia/Course/LinearAlgebra/sec3-1.pdf] 를 참고하였습니다.

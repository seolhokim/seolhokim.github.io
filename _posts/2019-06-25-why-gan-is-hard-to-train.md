---
layout: post
title:  "gan 학습이 어려운 이유 in cs230"
subtitle:   ""
categories: deeplearning
tags: etc
---

generator는 x -> x* (x는 noise input, x* 는 generated result) 과정을 진행한다.

propagatation 전 과정이 linear하다고 편차없음을 가정.

$$ x^{*} = x + \epsilon w^T $$

정도로 표현할 수 있으며, discriminator는 generator의 값을 가지고 학습하게 되므로,

discriminator에 들어가는 input은 $$ \hat{y} $$ 는 $$ \hat{y} = wx^* $$ 로 나타낼 수 있다.

그렇다면 $$ \hat{y^*} = wx^* + \epsilon ww^T $$ 가 된다.

여기서 갖게되는 insight는 x의 dimension이 많을수록 더 큰 $$\hat{y^* } $$ 을 가지게되는 결과를 가진다.

또한, $$ww^T$$는 항상 양이므로, w가 크게되면, $$x^* !=x$$ 임이 자명해진다.

그러나 discriminator를 학습시키기위해선 어느정도의 크기가 있는 w가 필요하고, 파라독스에 빠지게된다.

non-linear하더라도, saturated 되지않도록 보통 linear한 구간에서 학습이 이루어지기 때문에, 딥러닝의 구조에서 gan이 학습이 잘안되는 이유를

이런 관점에서 설명 가능하다.


//$$ \frac {\partial{\hat{y}}} {\partial{x}} = w^T $$ 이고,



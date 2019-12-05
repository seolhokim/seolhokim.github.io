---
layout: post
title:  "auto diff library 에서의 loss 설계"
subtitle:   ""
categories: deeplearning
tags: etc
---

$$ loss = (\hat{y} - y)^2 $$
이 mse loss 는 자연스럽게 이 자체로 network의 loss이므로 그대로 코드상을 옮겨도 된다.

하지만 reinforcement learning에서의 loss는 간단한 policy gradient 방법에서의 loss function을 보면

$$ \pi_{\theta} $$ 에 따른 average value 는 $$ J(\theta) = \sum_{s \in S}{d_{\pi}(s)V_{\pi}(s)} $$  다음과 같이 나타낼 수 있고,

$$ J(\theta) = \sum_{s \in S}{d_{\pi}(s)\sum_{a \in A}{\pi_{\theta}\mathbb{R}_s^a }} $$ 변형 가능하다.R은 state action value로 나타낼 수 있다.

여기서 J를 증가시키는 방법을 강구해야하는데, 이는 Differentiable하게 모든 function을 설계했으므로, 미분을 통해 가능하다.

J를 미분해서, $$ \theta $$ 의 변화량에 따른 값을 loss 에 넣어주면 되므로, $$ \nabla _{\theta} $$ 를 loss 값에 넣어주지 않아도 되는 것이다.



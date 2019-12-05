---
layout: post
title:  "ddpg loss function 구현 팁"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---
actor 의 loss는 policy parameter 를 $$ \theta $$ 라 하였을 때, greedy 한 방법은, 모든 step에서 maximization을 해야하는데,
이는 사실 continuous action space에서는 불가능에 가깝다. 그래서 policy를 Q의 gradient 방향으로 update 해주는데, 이를 식으로 나타내면

$$ \nabla _{\theta}Q^{\mu^k}(s,\mu_{\theta}(s)) (1)$$

로 나타낼 수 있는데, $$ \theta $$ 의 변화량에 따른 Q의 변화량 이므로, $$ \theta $$가 단위극소량 변했을 때, $$ \mu $$의 변화가 생기고 이 변화량
에 대해 Q가 얼만큼 변했는지에 대한 값을 얻게 된다. 이 값을 통해 policy parameter를 iterative하게 update하면, $$ \theta $$ 는 Q의 direction을 따라
update되게 된다. Q는 value function으로 상승하는 값을 향해 policy를 update하게 되는 것이다.

(1)을 구현하기 위해선 parameter $$ \theta $$ 의 변화량의 따른 policy의 변화에 따른 Q function의 변화량을 구해야 하므로, 식으로는 chain rule을
쓰지만, 구현하기 위해선 정말 간단하게 
- mean(Q(state,mu(state))) 를 쓰면 되는 것이다.


## reference

- (Deterministic Policy Gradient Algorithms)[http://proceedings.mlr.press/v32/silver14.pdf]
- (Continuous control with deep reinforcement learning)[https://arxiv.org/abs/1509.02971]

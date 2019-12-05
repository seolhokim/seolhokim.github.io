---
layout: post
title:  "Sample Efficient Actor-Critic with Experience Replay(ACER) 구현하기"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

## ACER 구현을 위한 hyper parameter 정리

- replay buffer memory : 50000
- single learning rate
- entropy regularization : 0.001
- discount the reward (lambda) : 0.99
- update Period : every 20 steps
- importance weight truncation (c) : 10
- ~~trust region updating~~ i will use PPO
- replay ratio : 0,1,4,8 
- network : 
  - $$ \theta $$ : policy network
  - $$ \theta_v $$ : value network(action state network)
  - $$ \theta' $$ : target policy network
  - $$ \theta_v' $$ : target value network
  - $$ Q^{ret} $$ : Retrace action state network
  - ~~$$ \theta_a $$ : average action network~~
  
  
~~~ 
# pseudo code - algorithm 1
## global shared parameter $$ \theta $$ and $$ \theta_v $$ !
## ratio of replay $$ r $$ <- poisson lambda
call ACER on-policy, Algorithm 2.
n <- get from possion distribution
for i \in \{1, \cdot \cdot\cdot , n \} do
  Call ACER off-policy, Algorithm Algorithm 2.
End for
until episode end
~~~
↓
~~~ 
# Python - algorithm 1
training(on_policy = True)
n = np.random.poisson(r)
for i in range(n):
  training(on_policy = False)
~~~

![algorithm 2](/assets/img/acer_algorithm.PNG)

<script data-ad-client="ca-pub-9593774082048674" async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>

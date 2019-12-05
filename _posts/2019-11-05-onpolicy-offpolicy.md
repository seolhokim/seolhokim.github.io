---
layout: post
title:  "On Policy와 Off Policy의 차이"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

## On Policy

On policy는 same policy로 부터 학습을 한다. SARSA를 예로 들면, state action value는 nextstate $$ s' $$와 현재 policy의 따른 
action $$ a'' $$ 를 통해 업데이트 되게 된다. 이는 같은 정책으로 exploit하고 update하는 방법이라고 볼 수 있따.

## Off Policy

Off Policy는 다른 policy를 통해 학습을 한다. Q-learning을 예로 들면, Q-value를 next state $$ s' $$ 와 greedy action $$ a' $$ 에 의해 학습하게 된다.
이는 greedy한 다른 policy로 학습한다고 볼 수 있다. replay memory 를 사용하는 RL Algorithm도 off policy로 볼 수 있다. 
independent identically distributed training set을 구성하기 위함도 있지만, 이전의 policy로 부터 얻어진
trajectory들도 replay memory에 남아있기 때문에 off policy라고 볼 수 있다.


### 참조문헌

 Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction. Second edition, MIT Press, Cambridge, MA, 2018.

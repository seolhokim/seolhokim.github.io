---
layout: post
title:  "Deep Reinforcement Learning with Double Q-learning (Double Dqn) 논문 리뷰"
toc: true
categories: 
  - Reinforcement Learning 
tags: [Reinforcement Learning]
author:
  - Seolho Kim
math: true
---

## Abstract

Q-learning은 action value(action state value, Q-value)를 overestimate한다는 것이 잘 알려져있다. 하지만 그런 overestimation이 실전에서
흔한지, harmful한지, 어떻게 예방할 수 있는지 에 대해서는 잘 안알려져 있고, 이 논문은 거기에 대해 모두 예라고 답한다. 

Double DQN을 통해 overestimation을 줄일 뿐만 아니라 더 좋은 결과를 내는 것을 보이는 논문이다.

## Introduction 

Q-learning은 Abstract에서 말한 것 처럼 overestimate를 하는 경향이 있는데, 이는 estimated action values를 최대화 하는 과정 때문이다.
overestimation은 이전 연구에서의 생각보다 자주 발생한다. 배우는 동안은 당연한 일이다.

다음 question은 overestimation이 발생하면, 안좋은 영향을 끼치냐이다. 모든 action이 value가 높다면, 상대적으로 선호되는 action이 보존될것이고,
이 것들이 나쁜 결과를 안일으킬 것임을 기대할 수 있다. optimistic한 것은 exploration에도 좋은 영향을 끼친다. 하지만, 그렇지않다면 나쁜 영향을
끼친다.

## Background

$$ \theta _ {t+1} = \theta _ {t} + \alpha(Y_t^Q - Q(S_t,A_t;\theta_t)) \nabla _{\theta_t} Q(S_t,A_t;\theta_t) $$ (1)

$$ Y^Q_t \equiv R_{t+1} + \lambda \max_a{Q(s_{t+1},a; \theta_t)} $$ (2)

가 기본 식이고, 이를 좀더 분석해 보자면, Loss 는

$$ Loss = (Y^Q_t - Q(S_t,A_t;\theta_t))^2 $$ 

square error 로 계산되는데, 이를 $$\theta $$ 에 대해 미분해서 $$ \theta $$ 를 구하려하면, (1)과 같은 식이 나온다. 

(2) 는 Q-learning의 $$Q(s,a;\theta)$$를 업데이트 하기 위한 식이다.

### Deep Q Network

Deep Q network(DQN)은 multi-layer neural network를 통해 Q function을 근사한 network로 $$ \theta $$ 가 network의 parameter로 나타냈을 때, $$ Q(s, \cdot;\theta) $$로 나타낼 수 있다.

DQN은 experience replay(replay memory)와 target network를 사용하는데, target network의 parameter를 $$ \theta ^- $$ 로 나타내고, 학습을 하면서 inference한 network와 train하려는 network가 달라지는 문제를 해결하기 위해 본래 network 외에 target network를 만드는데(총 network가 2개고 처음에 같은 parameter로 initialization을 진행한다.) 로 (2) 를 추론하고, update는 본래 network를 시키는 방법이다. 이렇게 하면, 본래 네트워크는 고정된 target network에 의해 network를 update하게 된다. 정해진 step마다 target network의 parameter를 본래 network의 parameter로 update한다.

### Double Q-learning

$$ Y_{t}^{Q} = R_{t+1} + \lambda Q(S_{t+1}, argmax_a Q(s_{t+1},a;\theta _ t); \theta _ t) $$

일반 Q-learning을 나타낸다. selection과 estimation을 하나의 network를 통해 하는 것을 나타낸다.

(2)같은 행위는 action에 대해 select하고 evaluate하는데 같은 value를 사용하게 되므로 overestimate 하게 될 확률이 높다. 그렇기 때문에 여기서는 selection과 evaluation을 분리시켰다. 

Original Double Q-learing은 두개의 value function이 random하게 update된다. update마다 한 network가 greedy policy를 결정하면 다른 network가 value를 담당한다. 

그아래는 double DQN의 기본 식을 나타내는데, 

$$ Y_t^{DoubleQ} \equiv R_{t+1} + \lambda Q(S_{t+1}, argmax_a Q(s_{t+1},a;\theta _ t); \theta' _ t) $$(4)

이를 통해 selection은 $$ \theta $$ parameter를 가진 network를 통해, evaluation 은 $$ \theta ' $$ parameter의 network를 통해 이뤄진다.

(다른 리뷰를 하신분은 이를 반대로 하셨다. 하지만 Q를 먼저 select하고, evaluate을 한다는 관점에서 이게 맞는 것 같다고 생각했다.)


그리고, 이 selection과 estimation은 두 네트워크가 번갈아가면서 실행하게 된다.

### Overoptimism due to estimation errors

overestimation의 정량적인 양에 대해 다룸. Theorem 1. 에선 lower bound 에 대해 다룸. action dimension이 클수록 lower bound는 감소한다. 이후 실험은 Double DQN은 estimation error가 안생겼는데, single Q-learning은 생김을 보였다.
그뒤론 Figure 2에대한 설명이다.

### Double DQN

결과적으로 Double DQN은 DQN과 같은 방식으로 update한다. 그러나, 다음과 같은 표기로 바뀌는데 살펴보면

$$ Y_{t}^{DoubleDQN} \equiv R_{t+1} + \lambda Q(S_{t+1}, argmax_a Q(s_{t+1},a;\theta _{t} ); \theta^{-} _{t}) $$(4)

다른 network였던 $$ \theta ' _ t $$가 target nerwork $$ \theta _ t ^- $$ 로 바뀐다는 것이다. DQN처럼 target network는 periodic copy가 이뤄지고, 이 target network는 evaluation하는데 사용된다는 점에서 DQN의 맹점을 잘 잡았다고 생각했다.

뒤의 내용은 실험에 관한 내용임으로 생략한다.

## Discussion

- Q-learning이 왜 overoptimistic한지 보였다.
- atari games에서 이런 overestimation이 흔함을 보였다.
- Double Q-learning이 이러한 overoptimism을 성공적으로 줄임을 보였다.
- Double DQN을 제안했다.(without additional network)
- Double DQN이 효과적임을 보였다.(여기선 SOTA)

## References
- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
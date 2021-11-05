---
layout: post
title:  "Soft Actor-Critic: off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor 논문 리뷰 및 설명"
toc: true
categories: 
  - Reinforcement Learning 
tags: [Reinforcement Learning]
author:
  - Seolho Kim
math: true
---

## Abstract

Model-free deep RL algorithm은 여러 방면으로 사용되었으나, 아직도 두가지 큰 약점이 있다. sample의 complexity가 높다는 점과(sample efficiency가
떨어짐), brittle convergence properties(hyperparameter에 영향을 많이 받음) 이러한 점들이 현실에서의 applicability를 어렵게 한다.
여기서는 soft actor critic을 제안한다.(SAC는 actor가 expected reward를 maximize하면서, entropy또한 maximizing하는 off policy actor-critic deep RL이다.)
off-policy update와 함께, stochastic actor-critic formulation의 combining을 통해, 여러 on-policy, off-policy method의 performance를 넘겼다.

## 1. Introduction

Model-free deep RL의 real-world domain으로의 challenges는 다음과 같다.

- notoriously expensive sample complexity
- brittle to their hyperparameters

sample efficiency의 poor함을 만드는 원인 중 하나는 on-policy라는 것이다. TRPO,PPO,A3C 모두 samples은 사용된 후 다음에 다시 사용되지 않는다.
이 때문에, gradient steps과 step마다의 samples이 많이 필요한 것은 policy update의 complexity를 높여서 안좋은 효과를 낸다. 하지만, off-policy는 past 
experience를 재사용함으로, 일반적인 policy gradient formulation에 적용은 어려우나, Q-learning based algorithm에 적용된다. 불운하게도, 이러한
off-policy algorithm과 high dimensional, nonlinear function approximation with neural networks은 stability와 convergence에 있어 큰 challenge
를 보였다. 이 것은 continuous state와 action space에서 더 심화되는데, Q-learning에선 separate actor network가 주로 maximization을 실행한다.
이러한 작업은 sample-efficient 를 높였지만, brittle하다.

여기서는 continuous state, action space에서 efficient하고 stable한 algorithm을 어떻게 design하는지 말한다. 결론적으로, maximum entropy framework를 제안하는데, 아주 기본적인 objective(목적식) 에서 entropy maximization term을 사용한다. maximum entropy RL은 original objective가 temperature parameter에 의해 다시 복구가 되어도 RL의 objective를 변화시켰다. 더 중요한건, maximum entropy formulation이 exploration과 robustness에서 improvement를 준다는 것이다. maximum entropy policy는 estimation errors를 만나도 다양한 behaviors를 획득함으로 exploration를 효과적으로 했다. off policy와 on policy 둘다 쓰였지만, 아까 설명한 이유들로 인해 off-policy가 좀더 까다로워도 sample complexity 부분에서 더 좋았다.

여기서는 off-policy maximum entropy actor-critic algorithm을 제시하는데, soft actor-critic(SAC)라고 부르고 여기선 TD3와 비교했다.

## 2. Related Work

여기서 SAC는 세가지 key 요소가 있다.

- actor-critic architecture with separate policy and value function network (여러 방법이 있었으나 여기선 아예 separate해버린다.)
- off-policy formulation
- entropy maximization

여기선 이 아이디어들을 도출하기 위한 prior works들을 리뷰한다.

Actor-critic algorithm은 Policy iteration으로부터 시작한다.

Policy iteration은 policy evaluation과 policy improvement로 이루어져 있는데, 전자는 policy를 value function을 사용해 평가하고, 후자는 
value function을 통해 policy를 개선한다. 하지만 large-scale RL problem에선 impractical하고, 이 둘을 분리해서 쓴다.
policy를 actor라하고, value를 critic이라고 하는데, 많은 on-policy algorithm에선 entropy를 고려하는데 그냥 regularizer 정도로 사용한다.
on-policy는 stability하지만 poor sample complexity하다.

robustness를 유지하면서 sample efficiency를 increase하는 노력으로는 off-policy를 사용하고, higher order variance reudction techniques를 사용하는 방법으로. 그러나 완전한 off-policy algorithm을 사용하는 것이 더 efficiency를 가졌다. 유명한 off-policy actor critic 방법으론 DDPG가 있는데, Q estimator를 사용해 off-policy를 가능하게 했고, deterministic한 actor가 Q-function을 maximize한다. 그렇기에 이 algorithm은 actor-critic이고, q-learning을 사용하는 것으로 보인다.(off-policy) 하지만, 이 deterministicactor network와 q-function의 상호작용이 DDPG를 stabilize하기 어렵게하고 brittle하게 만들었다. 결과적으로 이는 DDPG가 high-dimensional task에 extend하기 어렵게 만들었고, 그래서 on-policy 계열이 더 좋은 결과를 내도록 만들었다. 그래서 여기서는 ※off-policy actor-critic training을 stocastic actor와 함께한다!※ 더 나아가 actor의 entropy를 maximize한다. 이런 방법을 통해 DDPG보다 더 좋은 결과를 낳았다. SVG(0)이라는 알고리즘과도 비슷한데, separate value network를 사용하지 않았다. 이렇게 separate한 방법은 more stable했다.

Maximum entropy RL은 expected return 과 함께, policy의 entropy exceptation을 optimize한다. 이러한 framework은 많은 곳에서 사용 됐다. IRL로 부터 optimal control 까지. guided policy search 에서는 maximum entropy distribution이 policy learning을 high-reward regions로 가도록 도움을 주는 역할을 했다. 더욱 최근에 몇몇 논문은 maximum entropy learning에 대해서 Q-learning과 policy gradient method 의 연결에 주목했다.
또한 soft Q-learning algorithm도 제안됐는데, value function과 actor network를 가졌다. 그래도 이것은 actor-critic algorithm은 아니었는데, Q-function이 optimal Q-function을 estimating하고, actor는 데이터 분포에 따르는 걸 제외하고, 직접적으로 Q-function에 영향을 주지 않는다. 그래서 actor network를 sample을 approximate하는데만 사용했다. 그래서 이런 method의 convergence는 sampler가 얼마나 posterior를 잘 approximate 하냐에 달려있었다. 대조적으로 여기에선 policy parameterization에 상관없이 잘 converge했다. 거기다가 이전의 method들은 DDPG를 못넘었지만 여기선 SOTA를 넘었다.


## 3. Preliminaries 

### 여기서 부턴 그냥 혼자 읽었는데 구현이 잘 안되서 다시 읽으려고 한다.

### 3.2 Maximum Entropy Reinforcement Learning
일반적인 RL은 reward의 expection의 합을 maximization을 하는 것이 목표가 되지만 여기서는 maximum entropy 를 최대화하는하는 것이 고려되었다.

- explore efficiency
- capture multiple near optimal behavior
- already observed improved exploration with this objective

## 4. From Soft Policy Iteration to Soft Actor-Critic

SAC는 policy iteration에서 부터 출발했는데, 여기서는 이 도출을 보이고, converge함을 보이고, practical algorithm을 내놓는다.

### 4.1 Derivation of Soft Policy iteration

가장 중요한 건

$$ V(s_t) = \mathbb{E}_{a_t~\pi}[Q(s_t,a_t) - log \pi(a_t|s_t)] (3)$$

이 식이다. entropy loss가 value function에 들어와 value를 구하게 되므로, action의 entropy를 최대한 높이는 방향이 value function 에 크게 작용하게 된다.

또한 중요한 점은, policy를 새로운 Q function에 대한 exponential의 KL divergence를 통해 구하게 된다. 

## References
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
---
layout: post
title:  "Agent57: Outperforming the Atari Human Benchmark 논문 리뷰"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

# agent57

## 0. Abstract

Atari game은 RL community에서 오랫동안 벤치마크가 되어왔습니다. 이 벤치마크를 통해 알고리즘의 전반적인 성능을 테스트하곤 했는데, 이전의 많은 연구에서 꽤 좋은 성능들을 이미 보여왔습니다. 하지만, 어떤 분야에서는 굉장히 성능이 떨어지는면들도 보였는데, 이 논문은 atari 모든 게임에서 일반인을 능가하는 성능을 보였습니다. 이러한 결과를 얻기 위해 policy를 exploration을 주로 하는 policy부터 exploitation를 주로 하는 policy까지 여러 set으로 구성했으며, training을 하며 우선순위를 두어 어느 policy를 선택할지 학습하는 메커니즘을 설명합니다. 또한, 좀더 안정적인 학습을 할 수 있도록 architecturing을 하는 방법을 소개합니다.

## 1.Introduction

Atari game에서 RL performance를 측정할 때, 사람과 비교한 normalized score를 사용하는데, 또 주로 그 중에서도 어느정도 reasonable하다 판단되는 57개의 game에 대해서   benchmark로 사용하고 있습니다. 하지만 지금까지 이 모든 57개의 게임에서 이 human normalized score(HNS)를 넘은 경우가 없었고, 지금까지의 state-of-the-art(SOTA)를 보자면 Model-based인 MuZero가 있고, Model-free인 R2D2는 각각 51개와 52개의 게임에서 HNS를 능가했습니다. 인간의 점수보다 10배이상 잘하는 경우도 있었지만, 그렇지 않은경우 learning이 완전하게 이뤄지지 않는 경우도 있었습니다. 그렇기 때문에 완전히 일반화된 알고리즘이라고 보기에는 어려웠습니다.

일반화된 알고리즘을 보이기 위해서는 다음과 같은 중요한 이슈를 타파할 수 있어야 합니다.
* 첫째로, long-term credit assignment 입니다. reward가 지연될수록, 어떠한 action이 좋거나 나쁜 reward을 유발시켰는지 정확하게 판별하기 어려워집니다. 이러한 경우는 Skiing과 Solaris 게임에서 볼 수 있는데, Skiing은 모든 gate를 통과하며 경사를 최대한 빨리 내려오는 게임인데, gate를 통과하지 않을때마다 마지막에 5초의 페널티가 부과되고, 이는 credit assignment를 어렵게 만듭니다.
* 둘째로, exploration입니다. Private Eye, Montezuma’s Revenge, Pitfall!, Venture는 exploration이 중요한 게임으로 여겨지는데, 이는 좋은 reward가 기대되지 않아도 계속 exploration을 해야 게임을 clear할 수 있기 때문입니다. 이 문제는 특히 high dimensional state space(이미지 데이터같은)에서 주로 발생합니다.

Exploration algorithm은 다음과 같은 세 가지 카테고리 안에 들어갑니다.
* randomized value function
* unsupervised policy learning
* intrinsic motivation
이외에도 handcraft feature를 사용하는 등의 방법은 있었으나, 이러한 연구들은 큰 성능 개선을 가져다주지 않았습니다. 

하지만 최근에 Never Give Up(NGU)에서, 이런 hard exploration이 필요로 하는 게임에서 엄청난 성능개선을 보여주었고, 이는 intrinsic reward를 적극적으로 사용합니다.

이는 한 에피소드 내의 short-term novelty와 에피소드들 사이의 long-term novelty에 대해 sensitive하고, exploring과 exploiting을 하는 policies들을 배웁니다. 그러나 이는 위에서 언급했던 MuZero나 R2D2처럼 일반적이지 않습니다. R2D2는 optimal score를 얻을때, NGU가 거의 random policy처럼 행동하기도 했습니다.

NGU의 단점은, learning progress에서 그 기여가 얼만큼이 됐든, 모든 policies들의 experience를 같은 양씩 모은다는 점입니다. 몇몇 게임은 exploration의 정도가 다르게 적용되어야 합니다. 

#공유 자원(network capacity와 data collection)을 할당한다.(조금 더 이해 필요)

이렇게 함으로써 성능이 최대화되었고, exploration 전략을 상황에 맞추게하여 NGU가 더 일반적인 agent가 되도록 하였습니다.

최근의 long-term credit assignment는 다음과 같이 두가지 분야로 분류됩니다.
gradients를 바르게 credit에 assign하는 방법
value와 target을 사용해 credit이 잘 assign됐는지 확인하는 방법
좀 더 심화된 방법으로, 두가지 접근법을 같이 사용합니다.

여기서는 long-term credit assignment problem을 
* 1.discount factor를 dynamic하게 조정함
* 2. time window하는동안 backprob을 증가시킴 (r2d2를 읽어보아야함)
이전에 연구에 비하면 굉장히 쉬운 접근법이지만 이를 효과적으로 사용하는 법을 찾았습니다. 최근의 많은 연구들은 이처럼 hyperparameter를 어떻게 dynamic하게 조정하는지에 대해 이루어 졌습니다. 여기서는 간단한 non-stationary multi-armed bandit을 사용해 episode return을 maximize하기 위해 exploration rate와 discount factor를 직접 조정하고, 이 정보를 value network에게 input처럼 전달합니다.

결과적으로, 이 논문의 contribution은 다음과 같습니다.
* state-action value function을 intrinsic reward와 extrinsic reward의 contribution을 분리하는 방법을 통해 training stability를 높였습니다.
* meta-controller(exploration rate와 discount factor로 이루어진) 가 policy를 선택하는 메커니즘을 소개합니다.
* 처음으로 atari 57게임에서 모두 human baseline을 넘는 것에 대해 설명합니다. 또한 r2d2에서 좀만 re-tuning하여 publish된 paper보다 좀더 long-term credit assignment에 강하게 만든 결과를 설명합니다.


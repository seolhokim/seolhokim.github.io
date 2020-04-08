---
layout: post
title:  "Never Give Up : Learning Directed Exploration Strategies 논문 리뷰"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

# Never Give Up : Learning Directed Exploration Strategies

## Abstract

이 논문은 exploration이 주가 되는 게임에서 강점을 보이는 논문입니다.
여기서 intrinsic reward를 이용하였는데, 이는 episodic reward와 inter-episodic reward로 나뉩니다. episodic reward는 k-nearest neihgbors를 
사용하여 한 episode내에서 얼마나 새로운지를 판단해 reward를 내리는 self-supervised 기법입니다. 그리고, Universal Value Function Approximators(
UVFA)를 이용해, exploration과 exploitation의 trade-off를 맞추며, 다양한 exploration polices(exploration을 위해 여러 policies를 가집니다.)를 배우도록 합니다. 이렇게 한 네트워크가 다른 정도의 
exploration과 exploitation을 함으로써, 효과적인 exploratory policies가 효과적인 exploitative polices를 생산한다는 것으로 부터 설명됩니다.

#여기서 policy는 network를 뜻하지 않습니다. 헷갈리지 않길 바랍니다.

## Introduction

Exploration에 대한 역사를 설명하는데, stochastic policies를 사용하여 dense reward scenarios에서 이를 해결한 paper부터, 최근에는 exploration을 
유도하기 위해 intrinsic reward를 사용함을 소개하며, 지금 state가 이전에 방문했던 state들과 얼마나 다른지를 사용해 어려운 exploration을 해결한 
paper들 또한 소개합니다. 하지만 이에 대해서도 근본적인 한계가 있음을 지적합니다. state에 대한 novelty가 사라지면, 
exploration이 더 필요하든 말든, 다시 그 state를 가려는 intrinsic reward가 현저히 줄어들어버리기 때문입니다. 

다른 방법으로는, prediction error를 사용하는 방법이 있습니다. 이 방법 역시 high cost이면서, 모든 환경에 일반화하기 어렵습니다.

이 논문의 메인 아이디어로는 exploration과 exploitation을 같은 network로 부터 함께 배우는 것입니다. 이는 exploitative policy는 extrinsic reward에 집중하면서도, exploratory policy는 올바른 방향으로 exploration을 유지한다는 뜻입니다. 이를 해결하기 위해 UVFA의 framework을 사용했고, 다양한 exploratory policies를 사용합니다.

여기서는 intrinsic reward를 두가지로 나눠 정의합니다.
* per-episode(Episodic) novelty
  * 이 novelty는 한 episode내에서의 novelty이므로, episode 전체로 봤을 땐, familiar한 state도 revisit하게 해줍니다.
  * 한 episode내의 방문한 state들을 encoding시켜 저장합니다.
  * 여기에서 사용하는 similarity(novelty의 반대)는 이전에 저장된 state과의 similarity입니다.
* life-long novelty
  * 이 novelty는 점진적으로 familiar한 state일수록 낮습니다.
  * Random Network Distillation error로 규정합니다.
이러한 두 가지 novelty의 결합은 complex task, high dimensional state space에서의 관측되지 않은 state에 대한 일관성과 episode 내와 episode사이에서의 일관성있는 exploration을 유지하도록 돕습니다.

이 논문의 contribution은 다음과 같습니다.
* exploration bonus를 정의해(episode내부와 episode간의 novelty로) training process내에서 exploration을 계속 유지하도록 합니다.
* 위에서는 그냥 exploration이라고 적었지만, 사실 seperate exploration을 합니다. 여러 parameter들을 가지고, exploration policy를 사용하는데, 이후에 설명하도록 하겠습니다.
* 어려운 exploration이 필요한 게임에서 기존 SOTA보다 좋은 성능을 보였습니다.






  

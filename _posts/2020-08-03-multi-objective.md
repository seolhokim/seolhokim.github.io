---
layout: post
title:  "A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation 논문 리뷰 및 설명"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---
한줄 리뷰 : background가 많이 필요한 논문이다. Hindsight Experience Replay를 이해하고 있으면, Algorithm은 HER의 변형이므로 생각보다 쉬울 수 있다.
# A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation

## Abstract

이 논문은 multi-objective reinforcement learning(MORL)에서, 상대적인 objective들간의 선형적인 중요도(선호)에 대해 해결한 논문입니다. 
Agent는 multi-objective에 대해서 상대적인 중요도에 대해 알고있지 않습니다. 이는 유저의 reward shaping을 통해 어느정도 해결할 수 있지만, 다른 중요도에 따른 결과를
 얻기 위해선 큰 소요가 발생하게 됩니다. 이 논문은 하나의 agent만으로 모든 중요도 공간을 탐색할 수 있는 알고리즘을 제시합니다.

## Introduction

![Fig1](/assets/img/multi-object-policy-adap-1.PNG)

위의 그림을 보면, 일기 예보와 네비게이션같은 여러 경우에서 같은 objective의 틀을 가지고 있다고 해도, 그 기준이 다릅니다. 이런 상대적인 중요도는 이전엔 스칼라 값의 보상으로 결국 환경 설계자의 도메인 지식등에 크게 영향을 받았습니다. 이러한 제한은 특정 상황에 원하는 policy를 얻기위해 여러번 재설계를 해야하는 등 많은 소요를 발생시킵니다. 

이에 대해 MORL은 두가지 장점이 있습니다. 
* 첫째로, 여러 objectives를 합쳐서 의도치않은 방향의 학습이 이루어지는 경우를 줄일 수 있습니다.
* 둘째로, 중요도를 쉽게 변경하거나 다른 업무에 적용할 수 있습니다.

그리고 MORL은 두가지 전략을 통해 발전해왔습니다.
* 첫째로 하나의 objective로 변환해 사용하는 것인데, 이는 RL algorithm을 바로 적용하여 최적화시킬 수 있습니다. 하지만 이는 평균적인 objective에 대해서만 배우게되어 특정 중요도에 있는 optimal에 대해 배울 수가 없습니다.
* 두번째로, 전체의 optimal policy의 set을 계산하는 것인데, 이는 많은 계산량에 의해 scalability가 한계가 있습니다. 이후에 설명하겠지만, Pareto front를 표현하는게 domain의 사이즈가 커질수록 기하급수적으로 어려워지기 때문입니다.

이 논문은 하나의 policy network를 가지고, 중요도의 전체 space를 다루는 방법을 설명하는데, 이는 유저가 원하는 중요도에 대해 손쉽게 제공할 수 있도록 할 수 있음을 알 수 있습니다. 그래서 여기서는 두가지 MORL의 문제를 해결하는데, 
* 첫째로, multi-objective에서의 Q-Learning MORL with linear preferences의 이론적인 수렴을 보입니다.
* 둘째로, MORL을 더 큰 영역으로 확장하기 위해 neural network의 효과적인 사용을 보입니다.

이 논문은 두가지의 핵심 아이디어가 있습니다.
* 첫째로, 중요도를 가지고 일반화하여 축소시킨 optimality operator가 유효했다는 점입니다.
* 둘째로, convex envelope형태의 Q-value를 중요도와 최적의 정책 사이에서 효과적으로 결합하게 됩니다.

여기서는 Hindsight Experience Replay와 homotopy optimization을 사용하고, 새로운 환경에 대해 어떻게 자동으로 중요도를 선정하는지 보입니다.



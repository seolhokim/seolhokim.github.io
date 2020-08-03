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

![Fig1](/assets/img/multi-object-policy-adap_1.PNG)

위의 그림을 보면, 일기 예보와 네비게이션같은 여러 경우에서 같은 objective의 틀을 가지고 있다고 해도, 그 기준이 다릅니다. 이런 상대적인 중요도는 이전엔 스칼라 값의 보상으로 결국 환경 설계자의 도메인 지식등에 크게 영향을 받았습니다. 이러한 제한은 특정 상황에 원하는 policy를 얻기위해 여러번 재설계를 해야하는 등 많은 소요를 발생시킵니다. 

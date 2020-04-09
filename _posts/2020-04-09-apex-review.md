---
layout: post
title:  "Distributed Prioritized Experience Replay 논문 리뷰 및 설명"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

#Distributed Prioritized Experience Replay

## Abstract

이 논문은 분산 강화학습에 대한 architecturing에 주목한 논문입니다. 여기서는 actor와 learner가 분리된 것이 특징인데, actor는 공유된 network를 
통해 environment와 interact하며(action을 취하며), 그 경험을 공유된 replay memory에 저장합니다. learner는 그러면 그 buffer의 data를 가지고, 
network를 업데이트 합니다. 이 때, priortized experience replay(PER)의 main idea에 따라, actor들이 수집한 data의 priority를 계산해 학습하게 됩니다.

## 1. introduction

이전의 주된 연구 경향은 한 machine에 대해서 성능을 개선하는 것을 보이는 것에 중점을 보였고, 
distributed RL분야에서도 연구했던 방향은 gradients를 업데이트하기 위해 computation을 parallelizing하는 방법을 시도했다면,
이 논문은 data를 더 만들어내고, priority를 이용하는 방법으로 scale up을 할 수 있는 기법을 사용한 것에 contribution이 있는 논문입니다. 
(주로 data-generation에 집중하였다고 합니다.)



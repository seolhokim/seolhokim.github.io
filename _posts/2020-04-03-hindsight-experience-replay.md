---
layout: post
title:  "Hindsight Experience Replay 논문 리뷰"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---


# Abstract
Sparse reward(feedback)은 RL에서 직면한 큰 문제중 하나입니다. 이 논문에서는 이를 해결하기 위해 Hindsight Experience Replay(HER)라는 technic을 소개합니다. 
이 technic은 sparse하거나 binary한 reward definition에서 효과적인데 off-policy algorithm과 결합해 사용할 수 있습니다.
이 논문에서는 주로 로봇팔에대한 experiments들로 실험 결과를 보였으며, 이전엔 해결이 잘 안되던 문제들을 해결하는 성과를 보였습니다.

# Introduction

이 논문의 key idea는 간단합니다. 

model-free RL의 학습은 사람의 학습과 많은 다른 점이 있지만, 여기서 주목한 차이점은 사람은 optimal이 아니었던 행동에 대해서도
 배운다는 점입니다. 예를 들면 만약 사람이 하키를해서 퍽(puck)을 골대로 날렸을 때, 퍽이 골대를 빗나가더라도 사람은 이런 점에서 배울 수 있습니다.
 하지만 강화 학습에서는 (골대 거리와의 차이 등등)이런 것이 세세하게 정해지지 않는다면 Agent는 그 행동에 대해 적절한 피드백을 
 받을 수 없습니다. 

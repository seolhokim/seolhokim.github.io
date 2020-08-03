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

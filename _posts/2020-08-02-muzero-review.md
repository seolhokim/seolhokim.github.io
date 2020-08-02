---
layout: post
title:  "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)논문 리뷰 및 설명"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

# Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model


## Abstract
주어진 환경에서 agent의 미래 policy를 계획하는 능력은 예전부터 주된 과제들 중 하나였고, Alphago와 같이 Tree-based planning이 이 도전적인 분야를 해결했습니다. 하지만 이는 완벽한 simulator가
주어져야만 했으며, 실제 환경에서는 그렇게 dynamics가 복잡하거나 안알려져 있는 경우가 많습니다. 이 연구는 그래서 MuZero라는 알고리즘을 연구했으며, 이는 학습된 모델과
tree-based search와의 결합으로 인해 dynamics의 정보가 없는 경우에서 큰 성공을 거두었습니다. MuZero는 planning에 연관된 reward, policy, value function 들을 iterative하게 예측하는
 모델을 학습합니다.
 
## introduction

model-free algorithm은 정확하고 정교하게 앞을 예측해야하는 경우에 SOTA보다 떨어지는 경향을 보입니다. 이논문에서는 Muzero라는 model-based algorithm을 소개하는데, 이는 AlphaZero의 
tree-based search방식과 search-based policy iteration을 가지고, learned model을 학습과정에 융합시켜서 학습합니다.

메인 아이디어는 다음과 같습니다. 
(1) model이 observation을 받은 뒤, hidden state로 변환합니다.
(2) hidden state는 이전의 hidden state와 model내에서 선택된 action(실제 action이 아닌)을 통해 update됩니다.
(3) 이 과정의 매 스텝에서 모델은 policy와 value function, reward를 예측하게 됩니다.

이 모델은 end-to-end로 학습되며, 단순히 세 가지(policy, value function,reward)의 예측값의 정확도를 올리도록 하나의 objective로 학습됩니다. 여기에는 observation을 다시 복원하기 위한
모든 것을 hidden state로 가져가야하는 직접적인 제약이나 요구사항이 없습니다. 단지 위의 세가지를 잘 나타낼 수 있도록 hidden state로 만들어 주면 되기 때문에, 쓸모없는 정보들이 간소화되면서
성능이 올라감을 알 수 있습니다.

## 


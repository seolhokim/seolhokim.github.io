---
layout: post
title:  "Distributed Prioritized Experience Replay 논문 리뷰 및 설명"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

# Distributed Prioritized Experience Replay

## Abstract

이 논문은 분산 강화학습에 대한 architecturing에 주목한 논문입니다. 여기서는 actor와 learner가 분리된 것이 특징인데, actor는 공유된 network를 
통해 environment와 interact하며(action을 취하며), 그 경험을 공유된 replay memory에 저장합니다. learner는 그러면 그 buffer의 data를 가지고, 
network를 업데이트 합니다. 이 때, priortized experience replay(PER)의 main idea에 따라, actor들이 수집한 data의 priority를 계산해 학습하게 됩니다.

## 1. introduction

이전의 주된 연구 경향은 한 machine에 대해서 성능을 개선하는 것을 보이는 것에 중점을 보였고, 
distributed RL분야에서도 연구했던 방향은 gradients를 업데이트하기 위해 computation을 parallelizing하는 방법을 시도했다면,
이 논문은 data를 더 만들어내고, priority를 이용하는 방법으로 scale up을 할 수 있는 기법을 사용한 것에 contribution이 있는 논문입니다. 
(주로 data-generation에 집중하였다고 합니다.)

여기서 사용한 architecture는 주로 DQN과DDPG의 기반을 두고 있고 Arcade Learning Environment라는 benchmark를 이용해 성능을 테스트하였습니다.

## 2. Background

### Distributed Stochastic Gradient Descent
* 이 기법은 지도학습계열에서 먼저 neural network의 학습을 빠르게 하기 위해 parallelizing하는 방식으로 사용되었습니다. 동기적과 비동기적으로 업데이트 하는 방법이 있는데, 둘다 효과적임을 보였고, 표준이 되어갔습니다. 이를 강화학습에서 비동기적으로 업데이트 하는 방식이 활성화되어 GA3C와 PAAC등이 사용되었습니다.

## 3. Our Contribution : Distributed Prioritized Experience Replay
이 논문에서는 PER을 distributed setting으로 확장시켰고, 높은 확장성을 갖는 방법임을 보입니다. 이런 방법을 Ape-X라고 명명하였습니다.

![Apex](/assets/img/apex_1.PNG)

![Apex](/assets/img/apex_2.PNG)
사실 기본 algorithm은 정말 간단합니다. 기존의 PER를 안보셨다고 해도, Actor와 Learner가 분리되어 각 Actor는 learner의 buffer에 Local buffer를 sampling해서 처리해 넣고, learner는 그 buffer를 이용해 update하는 형식입니다. 중요하게 볼점은, 
* 1. learner의 buffer는 학습후 비운다. 
* 2. actor는 local buffer를 가진다. 
* 3. PER에서는 sampling된 sample들만 priority를 update해줬는데 여기서는 actor들이 replay에 넣을 때 다시 계산하니, 추가적인 computation없이 큰 문제를 해결하였습니다. 

정도입니다.

data를 sharing하는 것이 gradient를 sharing하는 것보다, 기본적으로 data가 outdated속도가 느리고, algorithm이 off-policy에 강하기 때문에 장점이 많습니다.

또한 actor마다 여러 exploration polices들을 주는 등 exploration problem등에서 좋은 면을 보여줄 수 있는 장점들이 있습니다.

### 3.1 Ape-X DQN
* Ape-X는 여러 framework에 응용 될 수 있습니다. 그렇기에 여기서는 double Q-learning과 multi-step bootstrap targets을 learning algorithm으로 정했고, dueling network architecture를 선택하였습니다. 결과적으로 다음과 같은 수식으로 update를 합니다.

![Apex](/assets/img/apex_3.PNG)

보시는 것처럼 multi-step Q_learning에서 주로 correction(truncate등의)을 사용하는데, 여기선 잘되었다는 내용을 합니다.

### 3.2 Ape-X DPG

![Apex](/assets/img/apex_4.PNG)

DDPG의 algorithm과 3.1의 G_t를 구하는 식을 합치면 적용이 어렵지 않습니다.

## 4. Experiment
생략합니다.
## 5. Analysis
생략합니다.
## 6. Conclusion
생략합니다.

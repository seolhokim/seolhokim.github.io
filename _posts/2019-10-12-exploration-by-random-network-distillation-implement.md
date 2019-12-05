---
layout: post
title:  "Exploration by Random network Distillation 논문 구현"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

# Exploration by Random network Distillation 구현

## 초안

### 고려해야할 것
- ppo model은 intrinsic and extrinsic reward 를 둘다 output 내줘야함 (기존 action, reward 에서 하나 더 추가된 것)
- intrinsic extrinsic trajectories train data 만들 때, intrinsic reward는 non-episodic하게
- intrinsic reward는 돌리면서 그 표준편차로 나눠줘야함
- observation도 -5~5로 clipping
- crossentropy loss에 넣어주기

![Figure 2](/assets/img/rnd_algorithm.PNG)

- N은 agent 갯수
- $$ N_{opt} $$ 는 train할때 몇번 반복할 것인지
- K는 train하기 까지의 trajectories 길이
- M은 observation normalization을 하기 위한 초기 initial step 인데 보통 max min 다 받을 수 있으니까, 그걸로 초기화 하면됨

그다음 PPO 알고리즘과 동일하게 돌아가고 intrinsic error만 잘 loss에 더해주면 될것같다!

### 만들 것

- ~~train data making function~~
- ~~network~~

- ~~intrinsic observation normalization by running data and clipping~~
- ~~intrinsic reward normalization by running data~~
- ~~intrinsic reward needs to be made in non-episodic~~
- ~~extrinsic reward needs to be made in episodic~~
- ~~masking~~
- ~~entropy~~
- ~~multi rollouts~~


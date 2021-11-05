---
layout: post
title:  "Improving Playtesting Coverage via Curiousity Driven Reinforcement Learning Agents 논문 리뷰 및 설명"
toc: true
categories: 
  - Reinforcement Learning 
tags: [Reinforcement Learning, Exploration]
author:
  - Seolho Kim
math: true
---
## Abstract

현대 게임의 복잡도와 크기는 계속해서 증가하고 있습니다. 이 때문에 실제 플레이와 관련되어 테스트를 진행하는 일은 더욱 어려워지게 됩니다. 그렇기에 실제 
사람들이 테스터가 되어 직접 테스트를 하는 기존의 방법으로썬 개발주기를 늦추는 요인 중 하나가 됩니다.
이 논문은 그러한 점을 해결하기 위해 curiousity를 사용한 RL agent를 가지고 state coverage를 maximize하는 방법으로 3d 게임내의 맵 전체를 테스트합니다.


## Introduction
Curiousity를 통한 exploration 전략에는 다양한 방법이 있습니다. 이 논문은 간단히 count-based exploration 전략을 사용하지만 이후에
심화시켜 mutual information이나 uncertainty를 이용하는 exploration 전략들을 사용하면 좀 더 재밌는 결과를 가져올 수 있습니다.

## Related Work
기존의 연구들은 low-dimensional environment에 대해 주로 연구가 되었습니다.

## Implementation
### A. Environment 
![playtesting_rl_0](/assets/img/playtesting_rl_0.PNG)
environment는 다음과 같은 500m x 500m x 50m 크기의 3d environment에서 실험이 이루어집니다. 자세한 환경에 대한 내용은 다음의 유튜브 영상을 통해 확인 가능합니다.


[environment youtube video](https://www.youtube.com/watch?v=cfm3R94FB_4)


agent의 action은 앞 뒤, 좌 우 움직임과 좌우 화면 전환의 continuous action과 jump의 discrete action이 혼합되어 사용됩니다.
(실제 적용을 위해서 응용하자면 user와 action space가 같아야 좀 더 정확한 디버깅이 될 수 있습니다.)

### B. Reinforcement Learning setup

Algorithm은 PPO를 사용했습니다.

agent의 observation space로 하나는 state-based의 coordinates등에 대한 vector 정보, vision array로 물체와의 거리등에 대한 정보 두가지로 실험하였습니다.

### C. Optimizing Coverage
이 논문에서는 actor 320개와 learner 1개를 둔 DPPO 구조의 architecture를 사용하였습니다.
이제 curiousity를 어떻게 정의했느냐인데, 위에서 언급했듯 count-based reward를 agent에게 주게됩니다. 이 때, 3d environment의 coordinates에 대한 count
를 통해 reward를 계산해야하는데, coordinates의 continuous한 특성으로 인해, threshold $$ \tau $$를 통해 discrete하게 만들어서 count합니다.

reward는 다음과 같이 정의 됩니다.

$$ R_t = R_{\mathrm{max}} * [ 1 - \frac{\mathrm{N_i}}{\mathrm{max_counter}}] $$

### D. Reset logic
initialization을 할 때, agent의 위치는 처음엔 중앙 근처에서 생성시키지만, 학습이 진행될수록, count를 세던 buffer의 위치에서 땅에 붙어있는 상황에 대해서
생성을 합니다.

### E. Collecting and visualizing data
생성된 trajectories에서 좀 더 주목할만한 결과를 뽑았던 방법에 대해 뒤에서 얘기하게 됩니다.

## Results

### A. Exploration performance and map coverage
random policy에 비해 확실히 높은 coverage를 보임을 볼 수 있습니다. observation space로 vision array를 사용했던 agent가 조금 더 좋은 성능을 보입니다. 
이 때, Fig. 4(a)를 보면, 실제 플레이어가 도달하지 말아야할 곳에 agent가 도달하기도 했는데 이들을 어떻게 visualization했는지에 대해 B에서 설명합니다.

### B. Exploration boundaries and regions of interest
actor들에 의해 너무나도 많은 trajoctories가 생성되고 이를 추려서 visualization하기 위해 이 논문은 다음과 같은 방법을 사용합니다.
Exploration boundary(EB)와 Regions of interst(ROIs)를 정한 뒤 이를 넘는 행위가 발생되거나, 한 목적지를 이전과 많이 다른 trajectory를 가지고 접근하면 이를 기록하게 됩니다.
EB는 map의 boundary정도로, ROIs는 EB내에 있지만 여기를 지나는 trajectories를 수집하고 싶은 points를 일컬어 말합니다.

### C. Connectivity graph
지금의 state는 단순 discrete point cloud로 볼 수 있는데, 이를 좀더 응용하여 graph를 만들어 활용 가능합니다. 예를 들어, 두 지점간의 path를 찾거나, 지형간의 유사성을 가지고 군집화를
 하는 등의 응용이 가능합니다.
### D. Termination states
players가 맵에서 움직이지 못하거나 하는 현상들을 찾기위해 agents의 끝난 지점들을 기록하고 이를 visualization하여 stuck하는 곳을 찾아낼 수 있습니다.

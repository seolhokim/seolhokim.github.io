---
layout: post
title:  "On Policy와 Off Policy의 차이"
toc: true
categories: 
  - Reinforcement Learning 
tags: [Reinforcement Learning]
author:
  - Seolho Kim
math: true
---

## On Policy

On policy는 behavior policy와 target policy가 같습니다. 즉 현재 행동하는 policy를 그대로 update할 목적으로 환경을 탐색합니다. 현재 policy를 통해 얻은 trajectories를 가지고 policy를 update를 하기 때문에 얻어지는 state distribution자체가 현재 policy에 의존적이게 됩니다. 그렇기 때문에, data dependent합니다. 이는 local optimal에 수렴할 수 있는 가능성을 만듭니다. 또한 한번 policy를 update한 후, 그 이전의 trajectories는 현재의 policy와 다르기 때문에 더이상 쓸 수 없습니다. 하지만 주로 update할 action selection이 stochastic하기 때문에, exploration strategy에서 off policy보다 편할 수 있습니다.(하지만 exploration을 통해 얻어진 trajectories 조차 data dependent하기때문에 한계라고 생각합니다.)

## Off Policy

Off policy는 behavior policy와 target policy가 다릅니다. 현재 행동하는 policies와 update할 policy가 달라도 된다는 뜻입니다. 이는 target policy와 behavior policy에 의한 distribution차이를 Importance Sampling(IS)을 이용해 해결하거나 target policy의 action selection을 주로 max의 연산으로 deterministic하게 취함으로써 해결합니다. 어떻게 action selection을 통해 distribution 차이를 극복하냐가 예전에 가졌던 의문이었는데, 생각해보면 간단했습니다. approximated action value function $$ \hat{Q} $$ 가 있을때, 한 state $$s_t$$에서의 $$\hat{Q}(s,a)$$의 더 정확한 값은 $$r(s,a)+\max_{a'}\hat{Q}(s',a')$$ 이기 때문에 업데이트 할 수 있는 것 입니다. 이는 target policy의 action을 제한시켰기 때문에, local optimal에 빠지지 않기 위해 exploration strategy가 필요합니다. 그중 가장 간단하게는 epsilon-greedy를 사용하지만, 이는 비효율적이기 때문에, 또 다른 많은 exploration 전략이 존재합니다.


### 참조문헌

 Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction. Second edition, MIT Press, Cambridge, MA, 2018.

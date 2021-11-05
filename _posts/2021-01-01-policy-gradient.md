---
layout: post
title:  "Policy Gradient Methods for Reinforcement Learning with Function Approximation 논문 리뷰"
toc: true
categories: 
  - Reinforcement Learning 
tags: [Reinforcement Learning, Policy-based]
author:
  - Seolho Kim
math: true
---
## Background 
- Value-Based 접근방식이 주를 이뤘었는데, Policy를 또 optimize할 필요없이 approximated value function에서 어떤 state에서 value function을 최대화 하는 값만 내놓는 것이 Policy로 정하면 됐기 때문에 비교적으로 간단하면서 좋은 성능을 보였다.
- 하지만 이는 두가지 한계점이 있는데,
  - 첫째로, policy가 value function의 max만을 취하므로, policy가 한 value function에 대해 deterministic하다는 점이다. 이는 exploration이 어렵게하고, stochastic해야만 하는 문제에 대해 근본적으로 해결할 수 없음을 뜻한다.
  - 둘째로, 위에서 설명했듯 Policy가 max만을 취해, value function의 사소한 변화가 Policy에 큰변화를 이끌 수 있다는 점이다. 이러한 큰 변화는 value-based method의 수렴을 어렵게 만드는 요소중에 하나로, 실험적으로 간단한 문제에서 value-based metohd가 수렴하지 못하는 점들을 본 이전 연구들이 있다.
## Description
- 이 논문에서는 한 deterministic하게 고정한 policy의 value를 계산할 수 있는 Value function을 approximation함과 동시에 independent policy function도 계산하겠다는 논문이다. 그 policy에 대한 update는 average reward per step $$\rho$$에 대해 maximize하는 방향으로 policy의 parameter $$\theta$$를 gradient ascent를 진행한다. $$\Delta \theta \approx \alpha \frac{\partial\rho}{\partial\theta}$$
  - average reward per step을 approximate한 unbiased function $$f$$를 train하고, 이는 REINFORCE algorithm에서 reward만가지고 학습한 것과는 다르다.
  - 또한, 완전히 처음인 접근은 아니지만, policy gradient에서 Neural Network(gradient differentiable function)를 이용해 수렴할 수 있음을 보여준 첫 논문이다.
- 이 논문에서 살펴봐야 할 점은 다음과 같다.
  - $$\Delta \theta \approx \alpha \frac{\partial\rho}{\partial\theta}$$를 통해 policy gradient를 update할 것임에서 그렇다면 $$\frac{\partial \rho}{\partial \theta}$$는 어떻게 정의할 것이고, 그렇게 정의했을 시 어떻게 값을 업데이트 할 수 있는지 이다.
    - 첫째로, $$\rho$$는 위에서 말한 정의와 같이 average reward per step이다. 이것의 최대화가 policy gradient의 주 목적이고, 이는 다음과 같이 정의할 수 있다.

      $$\rho(\pi) = \lim_{t \rightarrow \infty} \mathbb{E}\{r_1+r_2+r_3+...+r_n|\pi\} = \sum_{s}{d^{\pi}(s)\sum_a{\pi(s,a)\mathcal{R}^s_a}}$$

      $$d^{\pi}(s)$$는 policy $$\pi$$를 따를시, 시간이 무한이 흘러감에 따라 state $$s$$이 등장할 확률을 나타낸다.  그렇기 때문에 맨 오른쪽 식을 해석하면, 모든 state에 대해 policy가 어떤 행동을 할 확률과 그 행동에 대한 reward를 곱한 식이된다.

      - 여기서 중요한게, reward를 gamma없이, 다음처럼 전체평균에 의해 빼준 값을 사용한다. 즉, 실제 계산에 사용하는 reward는 $$r'_t = r_t -\rho(\pi)$$ 로 정의된다고 볼 수 있다.

        $$Q^\pi(s,a) = \sum_{t=1}^\infty\mathbb{E}\{r_t-\rho(\pi)|s_0=s,a_0=a,\pi\}$$

      그렇다면, $$\frac{\partial\rho}{\partial\theta}$$는 어떻게 될 것인가? policy $$\pi$$에 의한 value function을 $$V$$이라 할 때, $$\frac{\partial V^\pi(s)}{\partial\theta} = \frac{\partial}{\partial \theta}\sum_a{\pi(s,a)Q^\pi(s,a})$$이 정의 이고, 이를 미분하면 다음과 같다.

      $$\frac{\partial V^\pi(s)}{\partial\theta} =\sum_a\left [ \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a) \frac{\partial Q^\pi(s,a)}{\partial\theta} \right ]$$

      Q와 V의 관계는 다음과 같기 때문에 치환가능하다. $$V(s) = \mathcal{R}^a_s - \rho(\pi)+\sum_{s'}\mathcal{P}^a_{ss'}v^\pi(s')$$

      $$=\sum_a\left [ \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a)\frac{\partial}{\partial \theta }\left[\mathcal{R}^a_s - \rho(\pi)+\sum_{s'}\mathcal{P}^a_{ss'}v^\pi(s') \right ]\right ]$$

      $$R$$은 $$\theta$$의 영향을 받지 않으므로 0이어서 소거된다.

      $$=\sum_a\left [ \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a)\left[ - \frac{\partial}{\partial \theta }\rho(\pi)+\sum_{s'}\mathcal{P}^a_{ss'}\frac{\partial}{\partial \theta }v^\pi(s') \right ]\right ]$$

      $$\rho$$항을 좌변으로넘기고 좌변을 우변으로 넘기면 다음과 같다.

      $$\frac{\partial\rho}{\partial\theta} = \sum_a{\left [ \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a) + \pi(s,a) \sum_{s'}{\mathcal{P}^a_{ss'}\frac{\partial V^\pi(s')}{\partial\theta}} \right ]} -\frac{\partial V^\pi(s)}{\partial \theta}$$

      양변에 $$\sum_s{d^\pi(s)}$$을 곱하면 다음과 같다.

      $$\sum_s{d^\pi(s)}\frac{\partial\rho}{\partial\theta} = \sum_s{d^\pi(s)}\sum_a{\left [ \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a) + \pi(s,a) \sum_{s'}{\mathcal{P}^a_{ss'}\frac{\partial V^\pi(s')}{\partial\theta}} \right ]} -\sum_s{d^\pi(s)}\frac{\partial V^\pi(s)}{\partial \theta}$$

      괄호를 제거한다.

      $$\sum_s{d^\pi(s)}\frac{\partial\rho}{\partial\theta} = \sum_s{d^\pi(s)}\sum_a{ \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a) + \sum_s{d^\pi(s)}\sum_a\pi(s,a) \sum_{s'}{\mathcal{P}^a_{ss'}\frac{\partial V^\pi(s')}{\partial\theta}} } -\sum_s{d^\pi(s)}\frac{\partial V^\pi(s)}{\partial \theta}$$

      $$\sum_s{d^\pi(s)}\sum_a\pi(s,a) \sum_{s'}{\mathcal{P}^a_{ss'}\frac{\partial V^\pi(s')}{\partial \theta}}$$는 한 state로부터 어떤 action을 할때 어느 state로 가는 확률의 합과 그 action들에 대한 합이므로, next state에 대한 stationary distribution이 되어 다음과 같이 쓸 수 있다. 

      $$\sum_s{d^\pi(s)}\frac{\partial\rho}{\partial\theta} = \sum_s{d^\pi(s)}\sum_a{ \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a) + \sum_{s'}{d^\pi(s')}{\frac{\partial V^\pi(s')}{\partial\theta}} } -\sum_s{d^\pi(s)}\frac{\partial V^\pi(s)}{\partial \theta}$$

      $$\sum_s{d^\pi(s)}\frac{\partial\rho}{\partial\theta} = \sum_s{d^\pi(s)}\sum_a{ \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a) }$$

      $$\sum_s{d^\pi(s)}\frac{\partial \rho}{\partial \theta}$$는 stationary distribution 의 전체 state에 대한 합 과 $$\frac{\partial \rho}{\partial \theta}$$의 곱인데, $$\frac{\partial \rho}{\partial \theta}$$는 stationary distribution에 영향받지 않으므로, stationary distribution의 합 1을 곱한 것과 같다.

      $$\frac{\partial\rho}{\partial\theta} = \sum_s{d^\pi(s)}\sum_a{ \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a) }$$

      최종적으로 다음과 같은 식이 완성 된다.

    - 둘째로 $$\rho$$를 다음과 같이 정의했을 때에 대한 증명을 한다. 주로 이렇게 reward를 잡기 때문에 이게 더욱 익숙한 증명일 것이다.

      $$\rho(\pi) = \mathbb{E}\left\{\sum_{t=1}^\infty{\gamma^{t-1}r_t|s_0,\pi} \right\}, Q^\pi(s,a)  = \mathbb{E}\left\{ \sum_{t=1}^\infty{\gamma^{k-1}r_{t+k}|s_t=s,a_t=a,\pi}\right\}$$

      로 $$\rho$$를 정의한다면, 다시 $$\frac{\partial \rho}{\partial \theta}$$를 정의해보겠다.

      $$\frac{\partial V^\pi(s)}{\partial \theta} =\sum_a\left [ \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a) \frac{\partial Q^\pi(s,a)}{\partial\theta} \right ]$$

      $$\frac{\partial V^\pi(s)}{\partial \theta} =\sum_a\left [ \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a) \frac{\partial }{\partial\theta}\left[\mathcal{R}^a_s+\sum_{s'}{\gamma \mathcal{P}^a_{ss'}V^\pi}(s')\right] \right ]$$

      $$\frac{\partial V^\pi(s)}{\partial \theta} =\sum_a\left [ \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)+\pi(s,a) \left[\sum_{s'}{\gamma \mathcal{P}^a_{ss'}\frac{\partial V^\pi(s')}{\partial\theta}}\right] \right ]$$

      $$\frac{\partial V^\pi(s)}{\partial \theta}$$가 계속 나옴을 볼 수 있다. 생각해보면, 두번째 term은 policy 확률에 따라 이동된 뒤의 $$\gamma$$가 곱해진 $$\frac{\partial V^\pi(s')}{\partial \theta}$$인데, 이렇게 계속되면, state $$s$$에 대해 어느 state $$x$$로 $$k$$번 만에 갈확률 $$Pr(s \rightarrow x,k,\pi)$$에 대한$$\sum_a \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(s,a)$$를 얻어 식으로 정리했을때 다음과 같다.

      $$= \sum_x\sum^\infty_{k=0}{\gamma^k Pr(s \rightarrow x,k,\pi) \sum_a \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(x,a)}$$

      $$= \sum_s{d^\pi(s)\sum_a \frac{\partial\pi(s,a)}{\partial\theta}Q^\pi(x,a)}$$

      결국 이것이 정리되면, $$\frac{\partial}{\partial\theta}\mathbb{E}\left\{ \sum_{t=1}^\infty \gamma^{t-1}r_t \mid s_0,\pi\right\}$$이므로 $$\frac{\partial\rho}{\partial\theta}$$와 같아 정의하고자 했던 바가 된다.

      $$\frac{\partial\rho}{\partial\theta} = \sum_s{d^\pi(s)}\sum_a{ \frac{\partial \pi(s,a)}{\partial\theta}Q^\pi(s,a) }$$

      위에서 우리는 두가지 방식으로 정의한 미분 가능한 $$\rho$$를 미분가능한 $$\pi$$로 업데이트 할 수 있음을 보였다.

  - 2번 챕터 증명은 그렇다면, approximate한 $$Q$$에 대해서도 저 업데이트 식을 이용할 수 있느냐 이다. 그렇게 하기 위해선 다음과 같은 가정이 필요하다. stochastic policy $$\pi$$에 대해 어떠한 local optimal에 수렴한 parameter $$w$$를 가진 function $$f$$이 있을 때, 이는 policy에 의한 어떤 업데이트 할 변화량인 $$\Delta w$$가  0이 된다. 이를 다음과 같은 비례를 이용해 그 아래값이 0이 됨을 도출 할 수 있다.

    $$\Delta w \propto \frac{\partial}{\partial w }\left[ \hat{Q}^\pi(s_t,a_t) - f_w(s_t,a_t) \right]^2 \propto  \left[ \hat{Q}^\pi(s_t,a_t) - f_w(s_t,a_t) \right]\frac{\partial f_w(s_t,a_t)}{\partial w }$$

    이 때, $$\hat{Q}$$를 이용함은, $$R$$과 같은 unbiased $$Q$$를 이용함으로써 $$Q$$를 대체할 수 있다.

    $$ \sum_s{d^\pi(s)}\sum_a{ \frac{\partial \pi(s,a)}{\partial \theta} [Q^\pi(s,a) - f_w(s,a)]} = 0$$

    여기서 다음과 같은 조건을 걸어줘야한다.

    $$\frac{\partial f_w(s,a)}{\partial w}=\frac{\partial \pi(s,a)}{\partial \theta} \frac{1}{\pi(s,a)}$$

    그래야 위의 식을 다음과 같이 변형 가능하다.

    $$ \sum_s{d^\pi(s)}\sum_a{ \pi(s,a)\frac{\partial f_w(s,a)}{\partial w} [Q^\pi(s,a) - f_w(s,a)]} = 0$$

    $$ \sum_s{d^\pi(s)}\sum_a{ \pi(s,a)\frac{\partial f_w(s,a)}{\partial w} Q^\pi(s,a) } =  \sum_s{d^\pi(s)}\sum_a{ \pi(s,a)\frac{\partial f_w(s,a)}{\partial w}  f_w(s,a)} = 0$$

  - 3번 증명은 policy가 Gibbs distribution을 따르는 linear combination일 때, Q approximation function $$f$$를 어떻게 얻을 수 있는지를 보여준다.
  - 4번 챕터 증명은 이제 1,2번 증명을 통해, iterative하게 policy를 update하면, locally optimal policy를 얻을 수 있음을 보인다. 간단하기에 iteratation 식만 남겨둔다.

    $$w_k = w,s.t. \sum_s{d^{\pi_k}(s) \sum_a{\pi_k[Q^{\pi_k}(s,a)-f_w(s,a)]\frac{\partial f_w(s,a)}{\partial w}}}$$

    $$\theta_{k+1} = \theta_k+\alpha_k\sum_s{d^{\pi_k}(s)\sum_a \frac{\partial \pi_k(s,a)}{\partial \theta}f_{w_k}(s,a)}$$

## References
- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
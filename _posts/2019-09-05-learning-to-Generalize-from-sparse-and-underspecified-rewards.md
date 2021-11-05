---
layout: post
title:  "learning to Generalize from sparse and underspecified rewards 논문리뷰"
toc: true
categories: 
  - Reinforcement Learning 
tags: [Reinforcement Learning]
author:
  - Seolho Kim
math: true
---

## Abstract

NLP든 많은 문제에서 복잡한 input을 받지만, feedback은 성공 실패의 binary값을 받게되어, 행동에 대해 굉장히 명확하지 않은 rewards를 받게 되는 문제들이 있다.
이를 해결하기 위해 많은 성공적인 trajectories를 모으기 위한 KL divergence를 사용한 mode covering과, robust policy를 구하기 위한 mode seeking 단계를 제시한다.
그리고 Meta learning을 제시하는데, reward function을 정제한다. 이 정제된 reward function은 학습된 policy의 validation performance에 의해 구해진다.
Bayesian optimization을 월등히 능가했다.

## 1. Introduction

논문에서 정의한 sparse and underspecified rewards를 상세히 적어놨는데, 우연의 일치로 맞게되었지만 혹은 같은 reward를 받게 되었지만,
부정확한 reward에 대한 얘기를 한다. 이런 reward가 combinational하게 결합되었기 때문이라고 설명하고 이를 효과적으로 해결하기위해 Abstract에서 말했던
것 처럼 효과적인 탐험과 잘못된 trajectories를 없애 보편화된 행동을 하도록 만들어야 한다.

### exploration and generalization

효과적이고 원리에 입각한 exploration을 위해, 저자들은 disentangle combinatiorial search와 exploration from robust policy optimization을 제시한다.(combinatiorial한 trajectories를 풀고, robust한 policy를 사용해 exploration을 하겠다는 뜻)
특히 high entropy exploration을 위해 KL divergence 방향으로 mode covering을 하겠다고 한다는데 알고리즘을 봐야할 것 같다.
주어진 좋은 trajectories를 사용해 mode seeking 을 KL divergence 방향으로 한다는데 이것도 봐야알겠다.

이 논문에선 스스로 rich trajectory-level reward function을 발견할 수 있는지 보았는데 결론으로는 Meta-Learning과 Bayesian Optimization이 있다.
보완된 reward function을 optimize하기 위해 outer loop에서 일반적 효능을 극대화시킴! outer loop에서 (좋은 trajectories 들을 가지고 auxiliary reward function을 재학습시킨다는 뜻)

## 2. Formulation

### 2.1 Problem statement

a는 action trajectory

x는 complex input

y는 side information(label같은)

$$ R(a |x,y) \in {0,1} $$

위의 R은 $$R(a)$$로 간소화 해서 쓴다.



$$ \hat{a} \approx argmax_{a \in \mathcal{A}(x)} \pi(a|x) $$ 

$$\mathcal{A}$$은 x에 따른 combinatorial set이다. 

$$\mathcal{A}^{+} \equiv \{a \in \mathcal{A}(x) | R(a |x,y) = 1 \}$$

#### IML

$$ O_{IML} = \sum_{x \in \mathcal{D}} \frac {1}{|\mathcal{A}^{+}(x)|} \sum_{a^{+} \in \mathcal{A}(x)} \log{\pi (a^{+}|x)}  $$

자명하다. 

## References
- [learning to Generalize from sparse and underspecified rewards](https://arxiv.org/abs/1902.07198)
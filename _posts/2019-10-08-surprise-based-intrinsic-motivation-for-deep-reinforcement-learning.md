---
layout: post
title:  "Surprise-based intrinsic motivation for deep reinforcement learning 논문리뷰"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

exploration하는 과정이 비효율적이므로 어떤식으로 개선을 할 수 있을까 하다가 찾은 논문이다.

## Abstract

Exploration은 RL에서 key challenge인데, 현재까지 굉장히 simple heuristic한 방법으로 이루어졌었다.(e-greedy나 Gaussian noise 정도로) 여기서는
 효과적이고 수량적인 exploration 방법을 제시하는데 agent의 surprise(이후에 더 자세한 설명이나온다)라는 notion을 maximize하는 방법을 사용한다.
model은 intrinsic rewards를 형성해 transition probability의 kl-divergence 를 학습된 모델로 부터 배운다.

## Introduction

Exploration이 적절히 안되면, suboptimal을 얻게 된다. finite state 와 action 에 대한optimal exploration strategies에 대한 연구는
이미 이루어졌지만, obvious generalization혹은 guarantees가 없는 continuous space가 문제가 된다. 

task-independent(인퍼런스하는 과정과 independent 하다는 것 같다.)한 intrinsic motivation을 사용하면 좀 더 좋은 exploration을 얻게 된다. 
surprise를 실제 policy를 배우는 모델로 부터 probability distribution의 KL-divergence 로 정의했다. 이에 실용적으로 접근할 수 있는 approximations
 으로 두개가 있는데 intrinsic reward를 쓰거나 Baysian concept으로 접근함.
 
 

## Surprise incentives

policy update step은 surprise measure가 들어가게되면 성능과의 trade off 가 일어나게 되는데(바로 $$L(\theta)$$를 최적화 하는게 아닌, curiousity term이 추가 되기 때문에 당연하다. 여기서 (1)의 expression이 약간 이상했는데 보자면 

$$ min_{\phi} - \frac {1} {\vert D \vert} \sum_{(s,a,s') \in D} \log {P_{\phi}(s\vert s,a)} + \alpha f(\phi) (1) $$

여기서 transition probability(다음 상태로 갈 확률)를 왜 minimize 하려는지 이해가 안된다.

그래도 다음식을 이해하는데 무리가 없는데,

$$ max_{\pi}L(\pi) + \eta E_{s,a~\pi}[D_{KL}(P\Vert P_{\phi})[s,a]] (2)$$

agent가 가본 곳은 뒤의 텀이 작을 것이다. 왜냐면 이미 transition이 비슷하게 학습되었기 때문에 KL-divergence가 작다. 하지만 덜 갔던 곳(unfamiliar places)은 KL divergence가 높을 것이다. 그래서 이 KL divergence가 높은 곳을 갈 수 있도록 term을 추가해준 것이다.


~~(잘못 이해함)를 보면 optimal $$P$$ (transition probability) 에 현재 dataset of transition tuple D(replay memory같은)에서의 transition probability
$$ P_{\phi} $$ 를 KL-divergency의 loss를 주어서 같은 분포를 만드려고 한다. 이 loss term이 위에서의 surprise about experience를 잡기위해 나왔다. 그래서 reward term 도~~



$$ r'(s,a,s') = r(s,a,s') + \eta( \log{P(s'\vert s,a)} - \log{P_{\phi}(s'\vert s,a)}) (3)$$

이런식으로 나타내 reward function으로 부터 2의 max를 구할 수 있다. 하지만 당연하게도 transition probability $$ P$$ 는 unknown이어서 (2)를 근사하는 다른 solution을 아래 나타낸다.


#### 첫번째 method 
여기서 첫번째 method로는 KL-divergence를 cross entropy로 근사하는데, 

1 . $$H(P)$$가 finite하고 충분히 작다.

2 . $$P_{\phi}$$가 $$P$$와 꽤 다르다

이 두 가지를 만족한다면, cross-entropy를 다음과 같이 나타낼 수 있다.

$$H(P,P_{\phi})[s,a] = E_{s'~P(\cdot\vert s,a)}[-\log{P_{\phi}(s'\vert s,a)}] $$

그러면 $$D_{KL}$$식도 변형되는데, 다음과 같이 그냥 cross-entropy로 나타낼 수 있다.

$$ D_{KL}(P\Vert P_{\phi})[s,a] = H(P,P_{\phi})[s,a] - H(P)[s,a] $$

$$                          \approx H(P,P_{\phi})[s,a] (4)$$

reward 도 이렇게 바꿀 수 있다.

$$ r'(s,a,s') = r(s,a,s') - \eta \log{P_{\phi}(s'\vert s,a)} (5)$$

$$s'$$ 에 대한 놀람은($$ P_{\phi}$$와 context $$(s,a)$$로부터 얻어지는) intrinsic reward이다! 

#### 두 번째 method

다른 방법으로는 lower bound를 두는 방법인데, 

$$  D_{KL}(P\Vert P_{\phi})[s,a] =  D_{KL}(P\Vert P_{\phi'})[s,a] + E_{s'~P} [\log {\frac {P_{\phi'}(s'\vert s,a)}{P_{\phi}(s'\vert s,a)}}]$$

$$ \geq E_{s'~P} [\log {\frac {P_{\phi'}(s'\vert s,a)}{P_{\phi}(s'\vert s,a)}}] (6)$$

이다. 앞에 텀(KL-divergence)은 jensen 부등식에 의해 0보다 같거나 크다는게 증명되었으므로, 항상 같거나 크다

$$ r'(s,a,s') = r(s,a,s') + \eta( \log{P_{\phi'}(s'\vert s,a)} - \log{P_{\phi}(s'\vert s,a)}) (7)$$

(6)를 통해 (7) 로 나타낼 수 있고, (1)식을 통해 k 번 updates된 $$\phi'$$ 를 가지고 사용하게된다. (1)식이 optimization expression임은 자명했으나,
이걸 통해 어떻게 업데이트 하는지는 아직 모르겠다.

$$ r'(s,a,s') = r(s,a,s') + \eta( \log{P_{\phi_{t}}(s'\vert s,a)} - \log{P_{\phi_{t-k}}(s'\vert s,a)}) (8)$$

결국 (7) expression은 (8)으로 변환 가능하고, 결국 experiment에선 (5)와 (8) 두가지로 모두 실험했다.

## 3.1 Discussion
논문에선 reward를 두가지로 나눠 표현했는데 ((3)번 같은) 앞 term을 extrinsic reward, 뒤의 term을 intrinsic reward로 표현했다.

 intrinsic rewards 는 limit으로 갈 수록 없어지는게 이상적이다. 왜냐면 적당한 exploration이 일어나면 extrinsic reward에 대해서만 집중하는게 옳기 때문이다. 하지만 (5)번은 limit으로 가도 intrinsic reward가 0 으로 수렴하지 않기 때문에 poor 할 것이지만 이 논문에선 limit으로 가도 어차피 intrinsic rewards는 그냥 P의 entropy 이므로 extrinsic rewards를 잘 찾을 것이기 때문에 큰 걱정은 하지 않았다. (8)은 당연히 이런 issue 가 발생하지 않는다.
 
 그다음은 (8)식을 Bayesian surprise에 연결 시켰는데,
 
 $$ D_{KL}(P(\phi \vert h_t,a_t,\phi) \Vert P(\phi \vert h_t)) (8.1) $$ 로 표현했다.
 
 $$P(\phi \vert h_t)) $$ 는 $$\phi $$ 에 의한 distribution 이고, $$ h_t $$가 이전의 history다. 그러므로, (8.1) 은 이전 history로부터 $$a_t$$ 로인한 $$s_{t+1} $$가 일어났을 때의 transition probability다.
 
 바로 다음단계의 transition probability 이므로 Baysian하게 $$ P(\phi \vert h_t,a_t, s_{t+1})  $$ 를 구하면,
 
 $$  P(\phi \vert h_t,a_t, s_{t+1}) = \frac {P(\phi \vert h_t) P(s_{t+1} \vert h_t,a_t,\phi)} {E_{\phi ~ P(\sdot \vert h_t)[P(s_{t+1} \vert h_t,a_t,\phi)]}} $$
 
 로 나타낼 수 있다.
 
 결국, $$ E_{\phi ~ P_{t+1} \log{ P(S_{t+1} \vert h_t,a_t,\phi)}} - E_{\phi ~ P_{t} \log{ P(S_{t+1} \vert h_t,a_t,\phi)}} (9)$$ 로 표현될 수 있고,
 (8)은 Baysian은 아니지만, (8)도 (9)와 비슷한 정보를 가지고 있음을 알 수 있다.



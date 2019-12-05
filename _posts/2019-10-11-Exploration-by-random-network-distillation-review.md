---
layout: post
title:  "Exploration by Random network Distillation 논문 리뷰"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

한줄 리뷰 : Randomly Initialized network를 통해 forwarding 한번으로 intrinsic reward를 구할 수 있다는 게 신기했다@! 이걸로 supermariobro agent를 만들어볼 예정이다!

# EXPLORATION BY RANDOM NETWORK DISTILLATION

## Abstract

구현하기 쉽고 최소한의 overhead를 일으키는 exploration bonus method이다. 그 보너스는 observation의 feature를 predicting 하는 것인데 완전히 랜덤하게
고정되어 initialized 된 네트워크로 부터!(벌써 재밌을 것 같다.) Random Network Distillation(RND) bonus는 이 향상된 flexibility와 결합되어 
상당한 진전을 이루었다. 

## 1. Introduction 

RL은 Expected return(reward)를 최대화 하는게 목표이고, reward가 dense하고 easy to find할 때, 잘 작동한다. 하지만 현실의 task들의 reward는 sparse 하고 hard to find하다. 

최근의 Deep RL은 최근의 많은 어려운 문제들에 대해 병렬적으로 많이 복제한 환경에 대해 많은 양의 samples 이 해답인 것을 제시하는 것 처럼 보인다. 하지만 최근에는 이에 대하여 counts를 하거나 information gain같은 exploration methods를 내놓는다.

여기서는 꽤 simple to implement 한 exploration bonus를 설명하는데 high-dimensional observations에 대해 잘 작동한다고 한다. 오직 single forward passing 만이 필요하고, 이는 비슷한 상황에 대해 lower prediction errors를 갖는 것에 기반한다. 

많은 저자들이 말했듯이 input의 stochastic function이 문제가 되는데(완전한 랜덤같은 경우에 계속해서 큰 curiosity를 주게된다) 이를 효율적으로 개선하는 방법이 어렵다.

여기서는 이러한 stochasticity를 하기 위해 랜덤하게 initialize된 network를 이용하는 것이다.

## 2. Method

### 2.1 Exploration Bonuses

Exploration bonuses는 agent가 외부에서 받는 reward $$ e_t $$ 가 sparse 할 때, agent가 좀더 explore 할 수 있도록 돕는 방법이다. 우리는 이 $$ e_t $$ 를 새롭게 reward 를 time $$ t $$ 에서의 exploration bonus $$ i_t $$ 와 함께 $$ r_t = e_t + i_t $$ 로 정의하였다. 

여기서 agent가 novel한 state를 visit 하기 위해, novel state에 대해서 $$ i_t $$ 가 높아야하는게 바람직하다. count-based exploration methods 는 이러 한 방법의 bonuses 를 제공하는데, tabular 환경에서는 대략 $$ n_t $$ 에 반비례하게, non-tabular 환경에서는 similar to visited state에 대해 density model로 bonuses를 제공한다.

또다른 alternative는 prediction error 형태로 $$ i_t $$ 를 제공하는데, Curiosity-driven Exploration by Self-supervised Prediction를 보면 충분히 이해할 수 있을 것같다.


### 2.2 Random network Distillation

network는 두개로 구성된다. fixed randomly initialized target network 와 predictor network 이다.

target network : $$ f : \mathcal{O} \rightarrow \mathbb{R}^k $$ 

predictor network : $$ \hat{f} : \mathcal{O} \rightarrow \mathbb{R}^k $$

학습은  

$$MSE||\hat{f} (x;\theta) - f(x)||^2 $$

를 통해 이루어진다.(target은 non trainable이다)

그렇다면 predictor network는 target network에 distill될 것이고, 이미 train 된 state에 대해 적은 prediction error가 나올 것이다.

mnist에 대고 실험을 해봤는데 위와 같은 가설을 충족 시켜주는 결과가 나왔다. 0은 자주 본 state같은 상황이고, 다른 숫자들에 대해 실험해 보았는데, 

![Figure 2](/assets/img/rnd_mnist_test.PNG)

같은 결과를 통해 활용할 수 있음을 보였다. 그리고 이 방법은 target network를 predictor가 그대로 완벽히 따라할 가능성이 있을 수 있다고 생각되었지만 위의 실험결과를 보면 아닌걸 볼 수 있다.

#### 2.2.1 Sources of prediction errors

prediction errors 들은 다음과 같은 요인들에 의해 생긴다.

- 1. similar data가 부족한 경우
- 2. target function이 너무 stochasitc한 경우
- 3. target function의 복잡도를 fit하기 위한 class가 제한되거나, state의 필요 정보가 너무 부족할때
- 4. target function을 approximate하기위한 predictor를 찾는데 실패했을때

1번 요인은 허용 가능하지만, 2와 3은 피해야한다. 이런 상황을 RND는 정적인 network를 통해 deterministic하게 만들어버린다.

#### 2.2.2 Relation to uncertainty quantification
~~이부분은 이해도가 낮아서 다음에 다시 볼 예정이다.~~
RND prdiction error는 uncertainty quantification method와 비슷한데, 즉 data distribution $$ D = {x_i,y_i}_ {i} $$를 지닌 회귀 문제를 보자.
Baysian 상황에서 우리는 이전을 $$f_{\theta^{\ast}} $$ 로 맵핑된 $$ p(\theta^{\ast}) $$ 로 두고 결과에 대해 업데이트 한 후에 이전을 계산한다.(결과을 통해 baysian임을 가정해 이전의 네트워크를 계산해 deterministic하게 만든다는 것 같다) $$ \theta $$ 는 prediction error를 줄이기 위해 주어진다. $$ \mathcal{F} $$ 는 function $$ g_{\theta} = f_{\theta} + f_{\theta^{\ast}} $$ 에 의한 분포이고 그렇다면 prediction error 는 다음과 같은 정의로 나타낼 수 있다.

$$ \theta = argmin_{\theta} \mathbb{E}_{(x_i,y_i)~D} ||f_{\theta}(x_i) + f_{\theta^{\ast}}(x_i) - y_i ||^2 + \mathbb{R}(\theta) $$ 

여기서 특별한 케이스로 타겟 $$ y_i $$ 가 0이라면, 절댓값 내의 마지막 term이 사라지고, prior function을 distillation하는 것과 같다. 다른말로 distillation error는 uncertainty의 양과 같다!!

### 2.3 Combining intrinsic and extrinsic returns

~~이전의 실험들에서는 intrinsic rewards를 그저 better exploration을 위해 사용했다. 이런 rewards 는 game over같은 것을 의미 하지 않는다. 하지만 novel states를 찾는 것은~~

이전 실험들에서도 intrinsic reward는 non-episodic하게 다뤄졌다. game over를 당해도 intrinsic reward가 줄거나 하지 않는다. 이런 방법은 exploration을 하기 위한 자연스러운 방법인데, 미래의 novel state를 찾는 intrinsic return은 한 episode내에서도 일어날 수 있고, 여러 episodes에 spread 됐을 수도 있기 때문이다. 하지만 이를 episodic한 extrinsic reward $$ e_t $$ 와 결합하는 일은 obvious 하지 않다.여기에서의 solution은 그냥 reward 를 $$ R = R_E + R_I $$ 로 각각 두었고 그러므로 두개의 value network를 가진다. $$ V = V_E + V_I $$ 로.

### 2.4 Reward and Observartion Normalization

intrinsic reward는 표준편차로 나눠진다. 특히 random untrainable network가 있기 때문에 이를 더 섬세하게 다뤄야한다.

## 3 Experiments

원래 experiments는 눈으로 훑으면서 읽는데 여기서는 training에 필요한 여러 테크닉들이 많이 기술되어있어서 잘 보아야 한다.

combining episodic and non-episodic return 에 대해 다루고, discount factors에 대해 다루고 scaling등 rnn은 좋을줄알았는데 별로였다 이런 내용도 담겨있다.


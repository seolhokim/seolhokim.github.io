---
layout: post
title:  "Addressing Function Approximation Error in Actor-Critic Method (TD3) 논문 리뷰 및 설명"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

## Abstract 

function approximation error에 대해서, Q-learning같은 value-based method는 value를 overestimate하고, suboptimal에 봉착할 수 있다는게 잘 알려져 있다. 여기서는 이런 문제가 actor-critic에서도 존재하고, 이를 minimize하는 novel한 method를 소개한다.

이는 Double Q-learning을 기반으로 두고 있으며, critic pair에서 최소 value를 가져가 overestimation을 막는 방법을 사용했다. 또한, target network와 overestimation bias와의 연결을 그려보았는데, policy update를 지연시키면 성능이 좋아졌다.

## 1. Introduction

discrete action space 일때의 overestimation에 관한 issue는 많은 연구가 이루어졌지만, continuous control 에 대한 actor critic 방법에서의 비슷한 issue는 아직 덜 연구되었다.

overestimation bias는 undertrained estimator의 imprecision에 의해 당연하다. 이런 imprecis estimator에 의해 계속 업데이트 되다보면 error의 accumulation이 이뤄지고, 결국 suboptimal로 가거나 divergent behavior를 하게 된다. 

여기서는 overestimation property가 continuous control setting 에서의 DPG에서도 존재함을 주장하고,
discrete action setting에서의 해결책 Double DQN을 찾았다.(actor-critic 환경에서는 비효율적이지만)
training을 하며 Double DQN은 estimate와 value function을 분리했었다.이미 리뷰한 내용이 있으니 좀더 연구가 필요하면 다음 링크를 통해서 보면 된다. [https://seolhokim.github.io/deeplearning/2019/11/05/double-q-learning-review/]  불운하게도 actor-critic setting의 slow-changing으로 인해, target value와 current value의 similarity가 생겨 overestimation을 벗어나기 어렵다. 이는 Double Q learning 를 변형함으로써 해결할 수 있다. 독립적인 trained critic pair를 사용한다. 이는 less biased 됐지만 high variance를 가진다. 그렇기 때문에 clipping을 통해 이를 해결한다. 이는 overestimation bias가 true value estimation에 upper-bound 되어서 underestimated 된다. 이는 learning을 하며 propagated 되지 않는다.

여기선 overestimation bias에 대한 연결과 variance reduction을 위한 여러 요소들이 소개된다.
- target network 를 둔다. (variance reduction에 가장 중요한 역할)
- value와 policy를 연결하고, value estimator가 수렴할때 까지 policy update를 늦춘다. 
- novel regularization method를 소개하는데 더 variance를 줄이는데 효과적이다.

actor-critic에서 SOTA인 DDPG를 변형해 TD3 algorithm을 소개하는데, policy 와 value update에 대해서 function approximation error를 상호 고려한다.

## 2. Related Work

Overestimation bias와 high variance 에 대해 related work 에 대해 소개하고 여기에서의 연구에 대한 개략적인 내용이나옴. regularization을 value estimates 를 averaging 하는 방법으로 하여서 variance 줄이는 방법 나옴. 다른 연구에서의 variance 줄이는 내용나옴

## 3. Background

DPG와 Q-learning에 대해 설명함

## 4. Overestimation Bias

Q-learning 을 통해 discrete actions environment problem solving엔 max값을 취하므로 당연히 error 가 누적 되지만, actor-critic에서는 왜 생기는지 less clear하다.

4.1에선 basic assumption에서의 overestimation을 보이고, 4.2에선 clipped variant double q-learning이 overestimation을 줄임을 보여줄 것이다.

### 4.1 Overestimation Bias in Actor-Critic

actor critic 환경에서의 overestimation을 보이는데, policy는 DPG임을 가정한다.
policy parameters 는 $$ \phi $$로 나타낼 수 있고, approximate critic $$ Q_{\theta}(s,a) $$ 로 학습되는 policy network를 $$ \phi_{approx}$$라고 표기하고, true critic(이상적인)으로 학습되는 policy를 $$ \phi_{true} $$로 표기한다. 이 때, policy network update expression은 

$$ \phi_{approx} = \phi + \frac {\alpha} {Z_1} \mathbb{E}_{s_{p_{\pi}}} [\nabla_{\phi}\pi_{\phi}(s)\nabla _a Q_{\theta}(s,a)|_{a= \pi_{\phi}(s)}] $$ 

와

$$ \phi_{true} = \phi + \frac {\alpha} {Z_1} \mathbb{E}_{s_{p_{\pi}}} [\nabla_{\phi}\pi_{\phi}(s)\nabla _a Q^{\pi}(s,a)|_{a= \pi_{\phi}(s)}] $$

로 나타난다.

$$Z_1, Z_2 $$는 gradient의 normalization이고, (normalization이 없어도 좀더 stricter condition에선 발생한다) 
$$ \phi_{true} , \phi_{approx}$$ network 에 맞는 policy를 각각 $$ \pi_{apporx} $$ $$ \pi_{true} $$ 로 표기한다.

만약 gradient direction 방향이 local maximum이라면, 충분히 작은 $$epsilon_1 $$ 이 존재하고, $$ \alpha \leq \epsilon_1 $$ 이면, 

실제 정확한 policy보다 approximate critic expectation은  $$ \pi_{approx}(s) $$ 이 더 클 것이다. 이를 나타낸가 (5)식이다.

$$ \mathbb{E}[Q_{\theta}(s,\pi_{approx}(s))] \geq \mathbb{E}[Q_{\theta}(s,\pi_{true}(s))] (5)$$

반대로, $$ \epsilon_2 $$ 이 충분히 작고, $$ \alpha \leq \epsilon_2 $$라면, 실제 ideal state action value function에 대한 expectation 은  true policy가 더 클 것이다. 이를 나타낸 것이 (6)이다.

$$ \mathbb{E}[Q^{\pi}(s,\pi_{true}(s))] \geq \mathbb{E}[Q^{\pi}(s,\pi_{approx}(s))] (6)$$

value estimation이 true value보다 크다면, (= $$ \mathbb{E}[Q_{\theta}(s,\pi_{true}(s))] \leq \mathbb{E}[Q^{pi}(s,\pi_{true}(s))] $$ )
(5)와 (6)은 $$ \alpha < min(\epsilon_1,\epsilon_2)$$ 일 때, 

$$ \mathbb{E}[Q_{\theta}(s,\pi_{approx}(s))] \leq \mathbb{E}[Q^{pi}(s,\pi_{approx}(s))] (7) $$ 

가 된다.

update가 됨에 따라 overestimation이 줄어들지라도 error의 존재는 두가지 문제점을 야기한다. 
- overestimation은 uncheck 된 채로 넘어가며 more significant bias가 될 수 있다.
- inaccurate value estimation은 poor policy update를 만들 수 있다.

#### Does this theoretical overestimation occur in practice for SOTA methods?

DDPG에서 overestimation이 일어났음을 실험적으로 보았다.

### 4.2 Clipped Double Q-learning for Actor-Critic

다른 제안된 방법들은 actor-critic method에서 overestimation을 줄이는데 효과적이지 못했다.

Double Q-learning에서는 분리된 value function에 의한 greedy update가 이뤄진다.(2개의 separate value estimator를 가지고, 하나가 다른 하나를 update하는데 사용된다) 만약 두개의 value estimator들이 독립적이라면 그것들은 unbiased estimator가 될 수 있다. 

Double DQN에선, target network를 value estimator로 사용하고 policy는 current value network를 통해얻는다. actor-critic setting 에선 유사하게 current policy 를 통해 얻은 action을 통해 update한다. 이를 수식으로 나타내면 다음과 같다.

$$ y = r + \lambda Q_{\theta '}(s', \pi_{\phi}(s')) (8)$$

하지만 이런 actor-critic method는 update 속도가 느려서 current와 target network의 차이가 별로 없음을 알아냈다. 대신에 original Double Q-learning을 쓰면, 두 policy는 $$ (\pi_{\phi_1},\pi_{\phi_2}) $$ 로 critic은 $$ (Q_{\theta_1},Q_{\theta_2})$$ 로 나타내면, 다음과 같이 나타낼 수 있다.

$$ y_1 = r + \lambda Q_{\theta'_2}(s', \pi_{\phi_1}(s')) $$

$$ y_2 = r + \lambda Q_{\theta'_1}(s', \pi_{\phi_2}(s'))(9) $$

천천히보면, $$ Q_{\theta_1} $$ 는 actor $$\pi_{\phi_1} $$ 에 의해 selection이 이뤄지고,  $$ Q_{\theta_2} $$ 에 의해 estimation이 이뤄진다. $$ Q_{\theta_2} $$ 도 같은 원리로 이뤄진다.


Double DQN도 비슷한 overestimation을 겪어 Double Q-leanring이 좀더 effective했다.


$$ \pi_{\phi_1} $$ 는 $$ Q_{\theta_1} $$ 로 optimize 할 때, $$ Q_{\theta_1} $$ 의 target을  independent하게 estimate 했다면 bias를 피할 수 있다. 하지만 critic끼리 update할 때 결국 같은 replay buffer를 사용하기 때문에, 결과적으로 $$ Q_{\theta_2}(s,\pi_{\phi_1}(s)) > Q_{\theta_1}(s,\pi_{\phi_1}(s)) $$ 한 몇몇의 state가 발생할 수 밖에 없다. 결국 이로 인해 ($$y_1 $$ 은 $$ Q_{\theta_2}(s,\pi_{\theta_1}(s)) $$ 를 통해 update되므로!)  $$ Q_{\theta_1}(s,\pi_{\phi_1}(s)) $$는 점진적으로 overestimate 된다. 이문제를 해결하기 위해 여기선 less biased $$ Q_{\theta_2} $$로 $$ Q_{\theta_1} $$를 상계하는 방법을 소개한다.

$$ y_1 = r + \lambda \min_{i=1,2} Q_{\theta'_i}(s', \pi_{\phi_1}(s')) (10)$$

min을 통해 estimation의 최소값을 씀으로써, true estimation과 approximal estimation의 차이 $$ \epsilon $$ 의 upper-bound를 두게된다.

그러나 당연히 min 값을 쓰게 되므로 underestimation이 생길 수 있는데, 이는 뒤로 propagated되지 않는다는 것에서 좀 더 낫다고 보고있다.

구현단계에서 single actor를 사용해서 computational cost를 줄일 수 있다. 그럼으로 target을 하나만 사용하게 되고,  $$ Q_{\theta_2} > Q_{\theta_1} $$ 이라면 일반적인 update가 일어난다. $$ Q_{\theta_2} < Q_{\theta_1} $$ 면 overestimation이 일어 난 것이다.

##### 이부분은 살짝 해석이 어려워서 뒤를 이해하고와서 이해해야할것같다.
다음 benefit으로는 function approximation error를 random variable로 다룸으로써, 우리는 minimum operator가 높은 value들을 lower variance estimation error와 함께 제공해야함을 볼 수 있다. 
random variable의 분산이 증가함에 따라 random variable의 최소 기댓값의 감소에 따라서.(아직 이해 못함)

## 5.Addressing Variance
 
Section 4 에서, variance의 overestimation에 대한 영향을 다루는 동안, variance 자체를 다뤄야한다고 주장했는데, high variance는 learning speed를 줄이고, performance에 안좋은 영향을 끼친다.
이번 section에서 update를 하면서 error를 줄이는 것에 대한 강조를하고, target network와 estimation error 사이의 connection을 만들고, actor-critic learning procedure동안 variance를 줄이는 방법에 대해 주장한다.


### 5.1 Accumulating Error

value estimator가 future reward와 관련해 approximate했다면, variance는 비례해서 커질 것이고, discount factor가 클수록 variance는 급격히 커질 것이다. 거기다가 각각의 gradient update는 small mini-batch에 관한 error를 줄일 뿐이지, mini-batch 밖의 estimate error를 줄일 수 있다는 것은 보장되지 않는다고 주장한다.

### 5.2 Target Network and Delayed Policy Updates

이 section에서는 target network와 function approximation error 에 대해 검사해보는데, stable target 을 사용하는 것이 error의 증가를 막았다. 여기서 RL algorithm 을 designing 할 때, high variance estimates와 policy performance에 대해 고려하게 되는 insight 를 얻었다고 한다.

Target networks는 Deep RL에서 stability를 얻는 잘알려진 방법이다. approximator가 multiple gradient updates를 하면서, target network는 stable objective를 주는데, 이게 training data에 있어 큰 커버 범위를 보였다. target 없이는 wildly divergence 했다.

#### When do actor-critic methods fail to learn? 
target network가 없으면 high variance 를 보이고, 그게 diverge의 원인이 된다. 

만약 target network가 multiple update를 통해 error를 줄인다면, 그리고 high error state가 divergent behavior를 보인다면 policy network를 value network보다 lower frequency로 update하면 된다는 결론을 얻게된다. 여기서는 value error가 최대한 small하게 됐을 때 까지 policy update를 미룬다. 그래서 여기선 policy와 target network를 critic이 d번 학습한 후 학습하게 된다.
target network는 soft update를 하게 된다.

### 5.3 Target Policy Smoothing Regularization

deterministic policies는 overfit될 수 있다는 걱정이 있다. critic을 updating 할 때, deterministic policy를 사용하는 learning target 은 function approximation error의 inaccuracies에 민감하다.(target 의 variance를 높일 수 있다. ) 그래서 이는 regularization strategy를 통해 variance를 줄여야 함을 보여준다. 이는 target policy smoothing 인데, SARSA를 닮았다. 우리의 접근은 비슷한 action은 similar value를 가질 것이라는 생각으로 부터 시작한다. function approximation은 이것을 암시적으로 하는 반면에, 유사한 actions 사이의 관계는 training procedure를 바꿈으로써 명시적으로 바꿔질 수 있다. 여기서는 다음과 같이 target action 주변에 value를 fitting 한다.

$$ y = r + \mathbb{E}_{\epsilon}[Q_{\theta'}(s',\pi_{\phi'}(s')+\epsilon)] (13)$$

이는 value estimator의 smoothing의 장점이 있다. 실제로는 이 action에 대한 expectation을 small random noise를 target policy에 더하고 평균을 냄으로써 approximation을 했다. target update는 다음과 같다

$$ y = r + \lambda Q_{\theta'}(s',\pi_{\phi'}(s') + \epsilon)(14)$$

$$ \epsilon ~ clip(\mathbb{N}(0,\sigma),-c,c)$$

noise를 target이 original action과 close하기 위해 clip 했다. 이는 SARSA와 비슷한데, value estimator는 off-policy이고, noise가 target policy에 더해졌다는 것이다(policy와 독립적으로) value estimator는 noisy policy에 의해 traine 된다 .

## 6. Experiments

쉽게 읽을 수 있다.

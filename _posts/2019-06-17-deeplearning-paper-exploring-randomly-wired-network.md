---
layout: post
title:  "Exploring Randomly Wired Neural Networks for Image Recognition 1편 논문 리뷰"
subtitle:   "Exploring Randomly Wired Neural Networks for Image Recognition"
categories: deeplearning
tags: paper
---

이번 논문 주제로 정한 exploring randomly wired neural networks다.

여기선 random network를 생성할 때, ER, BA, WS 등의 방법을 사용하는데,

거기서 개선하는 방향에 대해 적혀져 있지 않다. (darts나 또 다른 방법들이 있긴해서 또 방향을 살짝 수정해야할 수도 있겠다)

나는 이렇게 랜덤 네트워크를 생성하고, 스스로 유동적으로 네트워크의 엣지를 생성하고 지우는 방향에 대해 논문을 쓸 예정이다.


1. stage마다 concatenation 하지않은 network 만들기.

2. attention 사용한 network 구성

3. label을 uniform하게 넣어 network의 activation이 떨어지는 edge 제거, 랜덤 edge 


## Abstract

image recognition에 대한 neural networks은 extensive manual design에 의해 발전함. (simple chain-like model부터, multiple wiring path를 통해). resnet이나 densenet같은 경우도 그때문이다.
Neural Architecture Search(NAS)는 이제 wiring과 opeation type의 결합된 최적화부분에 대해 exploring 하고 있지만, 경우의 수에 의해 제한되었고,
여전히 maual design이 필요하다. 그래서 이 논문에서는 connectivity patterns의 집합을 보다 다양하게 탐험했다. 이것을 위해
stochastic network generator를 먼저 제안했다. 여러 random generator는 benchmark에서 competitive accuracy를 보이는 다양한 random networks
를 생성했고, 이러한 결과를 통해 우리는 focusing on designing better network generators 해야함을 보였다. 

## 1.Introduction

우리가 지금 딥러닝이라고 부르는 것들은 connectionish의 cognitive science에 의한 접근에 의해 이루어졌다. how computational networks are wired
가 중요한 요소였다. 이런 관점에서 최근의 computer vision에서의 진화는 driven by moving from models with chain-like wiring에서 좀더
elaborate connection patterns으로 이루어 졌다. 예를들면 resnet이나 densenet 얘네들은 잘 wired 되었기 때문에 effective하다.

이런 트랜드에서 나아가 NAS는 나타났다 wiring pattern 과 operation을 jointly searching에 대한 유망한 방향으로. nas methods는 
암묵적으로 중요한 (아직 간과되는)network generator에 의존하면서 search에 주목한다. NAS network generator는 family of possible wiring patterns
를 정의한다. 그러나, resnet과 densenet 처럼 nas network generator도 hand designed다... the space of allowed wiring patterns은 제한된다.
이러한 관점에서 우리가 물어보는건 이러한 constraint를 완화하고 novel network generator를 디자인한다면 어떨까?

우리는 이 question을 explore했다. stochastic network generator에 의해 생성된 randomly wired neural networks들을 통해. 
bias를 줄이기 위해 generator를 ER, Ba, WS 모델을 사용했따. define complete networks를 위해 DAG구조를 만들었고, functional roles를 위해 simple mapping을 node에 했다.

 결과는 놀랍다. severl variants of these random generators는 competitive accuracy를 보였다. WS를 쓴 the best generators는 produce multiple networks that outperform or comparable to their fully manually designed counter-parts and the networks found by various neural architecture search methods. observed that the variance of aacuracy is low. yet there can be clear accuracy gaps between difference generators. ## 그렇기 때문에 초기 연결 방법의 개선이 필요할 것 같다.. 어떤 방법이 있을까. ## 이러한 observation은 network generator design 이 중요하다는 것을 제시한다.
 
 우리는 these randomly wired networks는 not prior free 라곤 못한다.
  ##그러면 output들을 그냥 바로 실제 output이랑도 연결해주면 어떨까?##
 많은 strong priors 들이 generator를 design할때 내재되었다. particular rule과 distribution에 의해. 각각의 random graph model은 certain probabilistic behaviors를 가진다. WS가 highly clustered 된것  처럼. 궁극적으로 generator design determined a probabilistic distribution over networks. the generator design은 내제한다 그런 prior들을. 이건 간과되어선 안된다.
 
  우리들의 work은 explore a direction orthogonal to concurrent work on random search for nas. these studies들은 보인다. random search 가 competitive하다. NAS search space 에서. NAS network generator 관점에서. 이러한 결과는 prior induced by the NAS generator design또한 좋은 모델을 만들 수 있다는 것을 시사한다. 반면에 우리들의 work는 NAS design을 establish하는 것 외에도 explore different random generator design.
  
  마지막으로, 우리는 이 논문에서는 새로운 transition을 제안한다. designing an individual network 에서 designing a network generator may be possible, 비슷하게 how our community가 변해왔는지, designing features 로부터 designing a network that learns features로. the importance of the designed network generator 은 암시한다. 머신러닝이 automated되었다는게 아니라 human design and prior가 network에서 network generator engineering으로 갔다는 것을. ## stage layer에서 attention을 쓰는게 좋을까?
 
 

---
layout: post
title:  "reinforcement learning why q-learning does not require importance sampling 큐러닝에서 importance sampling이 필요하지 않은 이유"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

~~importance sampling은 한 분포의 추정을 할 때, 다른 분포를 통해 추정을 하는 방법인데, td나 mc나 모두 trajectory를 이용하는데 q-learning은 
bootstrap된 action-value function의 maximum 값을 사용하기 때문에 할 필요가없다.~~

~~-->좀더 자세히~~


~~off policy 에서의 큰 근거가 되는 importance sampling은 q-learning에서는 본적이 없다. 왜일까? importance sampling을 왜쓰는가? 한 위에서 설명한 것 처럼 함수값을 구하기 어려운 한 분포를 다른 분포로 부터 구해서 기댓값을 구할 수 있는 방법이다. 그렇다면, Q-learning에서는 왜 이전의 분포로 부터 새로운 분포를 얻게되는데 importance weights가 필요 없는 것일까? 생각해보자. 이전의 policy 분포는 Q가 최대가 되는 action을 가졌을 것이다. 그렇다면, 새로 업데이트 된 분포는 현재 max Q를 가지도록 policy가 update되므로, 이전의 policy와 같은것이다. 그렇기에 분모와 분자가 같아 항상 1이다.~~

애초에 G^(pi/mu)를 구할 때, V를 사용하면, pi와 mu의 distribution에 따라 expectation을 구하기 위해 importance sampling을해줘야하는게 맞지만, Q-learning에서는 distribution을 고려할 필요가 없다.


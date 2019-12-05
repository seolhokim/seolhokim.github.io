---
layout: post
title:  "reinforcement learning 에서의 boostrapping이란?"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

cs234 등의 강의를 듣다보면 Bootstrapping에 대해 뭔가 지금까지 알았던 boostrap과 다른 느낌을 확 받게 된다.

operating system에선 ROM위에 올라가 운영체제를 실행시키던 Bootstrap.

statistics에선 표본을 복원추출해서 표본의 분포를 추정하는 방법으로 사용,

machine learning에서도 복원추출을 해서 샘플을 늘리는 방법이다.

reinforcement learning 에선 update step에서 값을 추정하기 위해 한 개 이상의 추정된 값을 사용하는 것을 말한다.

그렇기 때문에, dynamic programming에서 Policy에 따른 Value Function을 추정할 때, Computation 때문에 보통 역행렬을 사용하지않고, iterate를 통해
구하는데, 추정값을 통해 계속해서 value function을 update하므로, bootstrapping을 사용한다고 할 수 있다.

하지만 monte carlo에서는 하나의 trajectory를 지나면서 결과를 알게 되므로 추정값이 아니다. 그러므로 bootstrapping x

Temporal difference에서는 reward를 추측한 V를 사용하므로 bootstrap을 사용한다고 할 수 있다.

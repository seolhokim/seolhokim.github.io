---
layout: post
title:  "Never Give Up : Learning Directed Exploration Strategies 논문 리뷰"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

# Never Give Up : Learning Directed Exploration Strategies

## Abstract

이 논문은 exploration이 주가 되는 게임에서 강점을 보이는 논문입니다.
여기서 intrinsic reward를 이용하였는데, 이는 episodic reward와 inter-episodic reward로 나뉩니다. episodic reward는 k-nearest neihgbors를 
사용하여 한 episode내에서 얼마나 새로운지를 판단해 reward를 내리는 self-supervised 기법입니다. 그리고, Universal Value Function Approximators(
UVFA)를 이용해, exploration과 exploitation의 trade-off를 맞추며, 다양한 exploration polices(exploration을 위해 여러 policies를 가집니다.)를 배우도록 합니다. 이렇게 한 네트워크가 다른 정도의 
exploration과 exploitation을 함으로써, 효과적인 exploratory policies가 효과적인 exploitative polices를 생산한다는 것으로 부터 설명됩니다.

#여기서 policy는 network를 뜻하지 않습니다. 헷갈리지 않길 바랍니다.

## Introduction

Exploration에 대한 역사를 설명하는데, stochastic policies를 사용하여 dense reward scenarios에서 이를 해결한 paper부터, 최근에는 exploration을 
유도하기 위해 intrinsic reward를 사용함을 소개하며, 지금 state가 이전에 방문했던 state들과 얼마나 다른지를 사용해 어려운 exploration을 해결한 
paper들 또한 소개합니다. 하지만 이에 대해서도 근본적인 한계가 있음을 지적합니다. state에 대한 novelty가 사라지면, 
exploration이 더 필요하든 말든, 다시 그 state를 가려는 intrinsic reward가 현저히 줄어들어버리기 때문입니다. 

다른 방법으로는, prediction error를 사용하는 방법이 있습니다. 이 방법 역시 high cost이면서, 모든 환경에 일반화하기 어렵습니다.


  

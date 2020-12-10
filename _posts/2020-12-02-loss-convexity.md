---
layout: post
title:  "Convexity of network and corresponding parameter update"
subtitle:   ""
categories: mathematics
tags: etc
---
가끔 까먹는 이런 라이트한 주제도 다뤄보면 좋을것같아 개인 노션에 정리하던 것들을 가끔 풀어볼 예정이다.
1. **Background** 
    - Mean Squard Error, Cross Entropy Error는 regression과 classification에서 주로 쓰이는 loss들이다. 이는 loss를 convex하게 만들어주는 역할을 한다. 하지만, 우리가 업데이트하기 위한 parameters는 network내부의 non-linear activation function에 의해 non-convex인데, loss가 convex하다고, non-convex한 parameter를 가진 network를 최적화 시킬 수 있을까?
2. **Description**
    - 최적화 시키기 위한 parameter w_ij 에 대해서 Hessian은 항상 양이지 않을 수 있다. → non-convex
    - 이런 non-convex한 function을 network의 output에 convex한 loss(MSE,CEE)에 의해 update를 하게 된다. 과연 reasonable한가?
    - 최소한, convex loss의 global minimum을 찾았을 때, 유일하진않을 수 있더라도, parameter의 어떤 한 minimum을 찾을 수 있다.
    - 그렇기 때문에 adam등의 optimizer를 통해 non-convex한 parameter function을 update해 saddle point, local optimal등으로 부터 구출해내기 위함이다.

3. **Next Step**
    - sigmoid의 zigzag현상은 왜일어났더라?
4. **References**
    - [https://stats.stackexchange.com/questions/281240/why-is-the-cost-function-of-neural-networks-non-convex](https://stats.stackexchange.com/questions/281240/why-is-the-cost-function-of-neural-networks-non-convex)

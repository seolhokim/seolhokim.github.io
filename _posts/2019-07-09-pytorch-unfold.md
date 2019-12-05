---
layout: post
title:  "pytorch unfold에서의 차원"
subtitle:   ""
categories: deeplearning
tags: pytorch
---
  
pytorch 뿐만이 아니라 항상 dimension에 관한 작업을 하다보면 reshape이나 concat, split, matmul 등 가끔 헷갈리는
연산들이 있었는데, 이번에 Stand-Alone-Self-Attention 구현을 하면서 다차원 행렬의 연산에 대해 조금 이해하는
계기가 되었다.

예를 들면, 

~~~

import torch

test = torch.arange(0.,100).reshape((2,2,5,5))

~~~

이렇게 생성된 tensor가 존재할 때, 

unfolding을 하면, test.unfold(axis,number_of_components,strides) 식으로 하는데,

### unfold(0,2,1)

test.unfold(0,2,1) 을 두면, 0과 같은 dimension 위치에 있는 원소는 50이므로, 2개씩 묶게되면, [0,50]으로 묶인다.

이미 이해가 조금 있다면, strides가 어떤 값이든 같은 값이 나오는 걸 알 수 있는데, axis = 0 에 대해서 50개 모두
operation을 진행해야 하기 때문이다.

### unfold(1,2,1)

이제 axis=1에서의 원소 0 과 같은 dimension 위치에 있는 원소는 25이므로 위와 같이 진행이된다.

### unfold(2,5,1)

axis = 2 에서 원소 0과 같은 dimension 위치에 있는 원소는 5, 여기서 strides를 이렇게 저렇게 해보면 이해도가 높아질
것이다.

### unfold(3,1,1)

쓰다보니 이게 제일 먼저나와야했을 것 같은데, 이걸 만지는게 처음이면 제일 이해가 잘간다.

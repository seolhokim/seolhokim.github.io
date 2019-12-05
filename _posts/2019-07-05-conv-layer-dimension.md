---
layout: post
title:  "conv layer kernel을 구현할때 4-dimension인 이유"
subtitle:   ""
categories: deeplearning
tags: etc
---

https://stackoverflow.com/questions/46480699/why-are-my-keras-conv2d-kernels-3-dimensional

를 보니 dimension이 왜 4dimension인지 그냥 connecting을 위해서라고 간단히 넘어가는 것 같아서 사이트의 예를 들어 다시 좀더 정확하게 쓰면,
이전 feature map의 channel이 8개이기 때문에, kernel이 (kernel_width, kernel_height, before_channel) (width height순서가 헷갈린다) 그리고,
output으로 16개의 feature map을 뽑아낼 것이기 때문에, (kerenl_width, kernel_height, before_channel)의 shape인 kernel을 16개를 써서 4-dimension이
된다.

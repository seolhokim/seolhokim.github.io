---
layout: post
title:  "multi dimensional matrix multipulation. 다차원 행렬 곱셈 연구"
subtitle:   ""
categories: deeplearning
tags: etc
---

수학도 더 해야하는데 지금 다차원 곱에 대해 좀더 생각을 해봐야 겠다는 생각에 정리하려고 글을 쓴다.

말로 설명하는 것보다 직접 해보면서 직접 노트에 쓰면서 해보길 바란다.

여기에 있는게 어떻게 이런 shape이 나오고 값이 나오는지 알면 일단 남의 구현물을 이해하는데는 시행착오 조금 할 수 있어도 큰 문제는 없을 것이다.


~~~

import numpy as np
import tensorflow as tf
import torch
import keras.backend as K

test_1 = np.ones(200).reshape(5,4,10)
test_2 = np.ones(200) * 3
test_2 = test_2.reshape(5,4,10)

print(np.einsum("ijk,ibk -> ijb",test_1,test_2).shape)

print(np.einsum("ijk,abk -> ijab",test_1,test_2))

print(np.einsum("ijk,abk -> iajb",test_1,test_2))

test_1 = K.ones(200)
test_1 = K.reshape(test_1, (5,4,10))
test_2 = K.ones(200) * 3
test_2 = K.reshape(test_2, (5,4,10))

print(K.eval(K.batch_dot(test_1,test_2,axes=[2,2])).shape)


test_1 = K.ones(200)
test_1 = K.reshape(test_1, (5,2,2,10))
test_2 = K.ones(200) * 3
test_2 = K.reshape(test_2, (5,2,10,2))
print(K.eval(K.batch_dot(test_1,test_2,axes=[3,2])))


test_1 = np.ones(200).reshape(5,2,2,10)
test_2 = np.ones(200) * 3
test_2 = test_2.reshape(5,2,10,2)
print(np.einsum('ijkl,ijlm->ijkm',test_1,test_2))
~~~

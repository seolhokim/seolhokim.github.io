---
layout: post
title:  "dimension permute 에 대해.torch permute나 keras permute_dimensions"
subtitle:   ""
categories: deeplearning
tags: etc
---
multi dimensional matrix manipulation 처럼 이것도 해보면서 하면 금방 이해할 수 있다.

~~~

from keras import backend as K
import numpy as np
import torch

test_1 = (K.constant([x for x in range(200)]))
test_1 = (K.reshape(test_1,(2,5,2,10)))

print(K.eval(K.permute_dimensions(test_1,(0,3,2,1))))

test_1 = K.variable(np.array([x for x in range(4000)]))
test_1 = K.reshape(test_1,(10,2,5,40))

print(K.eval(K.batch_dot(K.permute_dimensions(test_1,(0,2,1,3)),K.permute_dimensions(test_1,(0,2,1,3)),axes=[3,3]))[0])

print(K.eval(tf.einsum("ijkl,izkl->ikjz",test_1,test_1))[0])
~~~

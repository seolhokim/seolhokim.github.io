---
layout: post
title:  "심심해서 SENet 테스트"
subtitle:   ""
categories: deeplearning
tags: etc
---

channel 단위로 squeeze 해서 excitation을 한다는 게 attention같은 느낌이 들어서 재밌었다.
channel을 그냥 fc에 projection해서 channel-wise dependencies를 capture하는데,
중간에 dimensionality-reduction된 layer를 넣어 non-linearity와 computation을 한꺼번에 잡았다.
inception module에서는 어떻게 썼고, resnet에선 어떻게 썼는지 등 보여줬는데, 어디다 써도
separable conv보다 computation은 높지만 channel dependencies를 구한 attention효과가 있기 때문에,
좋은 성능을 낼거라고 판단이 들었고 실제로 

http://dongyukang.com/CIFAR100-%ED%95%A9%EC%84%B1%EA%B3%B1-%EC%8B%A0%EA%B2%BD%EB%A7%9D-%ED%95%99%EC%8A%B5/

여기 모델에서 conv2d마다 넣어줬더니 3~4%p 정도 좋은 효과가 났다.(100epochs 기준)

~~~
class SEModel(Model):
    
    def __init__(self,channel,ratio):
        super().__init__()
        self.channel = channel
        self.ratio = ratio
        
    def call(self,inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = Dense(int(self.channel/self.ratio), activation = 'relu')(x)
        x = Dense(self.channel,activation = 'sigmoid')(x)
        x = Reshape((1,1,self.channel))(x)
        x = inputs * x
        return x
    
    def compute_output_shape(self,input_shape):
        return input_shape
~~~


---
layout: post
title:  "Keras conv2d layer 만들때 조심할 점"
subtitle:   ""
categories: deeplearning
tags: keras
---

지금 알았다는게 신기한게..

왜 계속 train해도 학습이 안되나 고민했는데, self.kernel 이 non-trainable했던 것 같다..

build에서 (kernel_size_height,kernel_size_width,input_channel,output_channel) shape으로 trainable weight로 만들어서 대신 넣어줬더니

다시 학습이 됐다... ㅠㅠ 왜 지금알았죠..?

bias 추가하려면 마찬가지로 build에서 output_channel의 weight 생성해서 add_bias넣어주면 깔끔하게 들어간다.

~~~
class Block(convolutional.Conv2D):
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 momentum=0.95,
                 **kwargs):
        super(Block, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
    def build(self, input_shape):
        channel_axis = -1
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape = kernel_shape, initializer= self.kernel_initializer,name='kernel',\
                                     regularizer = self.kernel_regularizer, constraint = self.kernel_constraint)
    def call(self,inputs):
        #x = K.relu(inputs)
        print(inputs)
        x = K.conv2d(inputs, self.kernel, strides = self.strides, padding=self.padding)
        print(x)
        #print("block ",x)
        #x = K.bias_add(x,self.bias)
        #x = BatchNormalization()(x)
        if self.activation is not None:
            return self.activation(x)
        return x
~~~

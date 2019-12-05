---
layout: post
title:  "multihead attention in cifar10"
subtitle:   ""
categories: deeplearning
tags: attention
---

cs234를 듣다가 너무 어려워서(Emma 교수님 말씀이 너무 빨라요 흑흑)

self attention 관련한 강화학습 논문을 쓸까 고민하고 있었는데 테스트용으로 self attention 구현을 해보았다.

https://keras.io/examples/cifar10_cnn/

에서 쓰인 cifar10 sample code에 덧입혔는데, 파라미터 수는 25% 줄였지만 성능은 거의 비슷하다. 하지만 overfitting이 심함.

~~~
class Attention(Layer):
    def __init__(self,head_num, head_unit):
        super().__init__()
        self.head_num = head_num
        self.head_unit = head_unit
        
        self.output_dimension = self.head_num * self.head_unit
        
    def build(self,input_shape):
        self.query = self.add_weight(name = 'query',shape = (input_shape[-1], self.output_dimension),\
                                     initializer = 'glorot_uniform',trainable = True)
        self.key = self.add_weight(name='key',shape = (input_shape[-1], self.output_dimension),\
                                  initializer = 'glorot_uniform', trainable = True)
        self.value = self.add_weight(name = 'value', shape = (input_shape[-1], self.output_dimension),\
                                    initializer = 'glorot_uniform', trainable = True)
        super().build(input_shape)
    
    def call(self,inputs):
        print('inputs.shape',inputs.shape)
        
        query = K.dot(inputs,self.query)
        key = K.dot(inputs,self.key)
        value = K.dot(inputs,self.value)
        query = K.reshape(query,(-1, self.head_num,self.head_unit))
        key = K.reshape(key,(-1, self.head_num,self.head_unit))
        value = K.reshape(value,(-1, self.head_num,self.head_unit)
        
        x = K.batch_dot(query,key,axes=[2,2])
        x = K.softmax(x)
        x = K.batch_dot(x,value,axes=[1,1])
        return x
    
    def compute_output_shape(self,input_shape):
        return input_shape[0], self.head_num, self.head_unit
~~~


~~~
inputs = Input(shape = x_train.shape[1:])

x = Conv2D(32,(3,3),padding='same',activation = 'relu')(inputs)
x = Conv2D(32,(3,3), activation = 'relu')(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64,(3,3),padding='same', activation = 'relu')(x)
x = Conv2D(64,(3,3), activation = 'relu')(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Attention(4,32)(x)
x = ReLU()(x)
x = GlobalAveragePooling1D()(x)

x = Dense(num_classes,activation = 'softmax')(x)

model = Model(inputs,x)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics =['acc'])
~~~

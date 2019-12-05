---
layout: post
title:  "Vanila Policy Grdient Keras"
subtitle:   ""
categories: deeplearning
tags: reinforcement
---

3dball agent 테스트를 위해 한번 만들어보았는데 성능개선이 잘안되서 잘못짰나 싶어 다시 개선할 예정

https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2 를 참고함

discrete하게 값을 뽑아내고있었음

이렇게되면 update를 할 수 없다. 

mu와 sigma output을받아 이값을 정규분포로 만들어 여기서 표본을 뽑아 행동을 해야했음.
~~~

import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers


class Agent(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[32, 32]):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__build_network(input_dim, output_dim, hidden_dims)
        self.__build_train_fn()

    def __build_network(self, input_dim, output_dim, hidden_dims=[32, 32]):
        self.X = layers.Input(shape=(input_dim,))
        net = self.X
        net = layers.BatchNormalization()(net)
        for h_dim in hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        net = layers.Activation("tanh")(net)
        #net = layers.Activation("softmax")(net)
        #net = layers.BatchNormalization()(net)

        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        discount_reward_placeholder = K.placeholder(shape=(None,self.output_dim),
                                                    name="discount_reward")
        action_prob = self.model.output
        log_action_prob = K.log(action_prob)
        
        loss = - log_action_prob * discount_reward_placeholder
        
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   #constraints=[],
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           #action_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    def get_action(self, state):
        shape = state.shape

        if len(shape) == 1:
            assert shape == (self.input_dim,), "{} != {}".format(shape, self.input_dim)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.input_dim), "{} != {}".format(shape, self.input_dim)

        else:
            raise TypeError("Wrong state shape is given: {}".format(state.shape))

        action_prob = self.model.predict(state)
        return action_prob
        
    def fit(self, S, R):
        discount_reward = compute_discounted_R(R)
        self.train_fn([S,discount_reward])

def compute_discounted_R(R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean() / discounted_r.std()

    return discounted_r
    
~~~

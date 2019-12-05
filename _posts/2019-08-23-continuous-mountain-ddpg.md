---
layout: post
title:  "MountainCarContinuous-v0 게임 keras tensorflow를 통한 ddpg 구현"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

웬만하면 혼자 구현하려 했는데, 

$$ \frac{1}{N} \sum_i \nabla_{a} Q(s,a \vert \theta ^Q) \vert_{s=s_i,a = \mu(s_i)} \nabla_{\theta^{\mu}} \mu(s \vert \theta{\mu}) \vert_{s_i} $$

부분을 어떻게 구현해야 할지 많이 고민했었다.

이 때, 앞의 항은 critic으로부터 tf.gradients를 사용해 actor에 K.function으로 끌고온다음 뒤의 항은 그냥 state에 대해 tf.gradients로 
action을 미분해 곱하면 될 것 같았는데,

일단 가장 star를 많이 받은 게시물을 보니 Q의 미분값을 가져온 상태에서 state에 대해 action의 gradient 를 구해 이 값을 그대로 loss로 썼다.

그러면 이게 감각적으로 이해는 됐지만 한번 실험을 해봐야겠다.

vanila policy gradient -> advantage policy gradient 둘다 수렴할 기미를 안보였었는데, 100 trials도 되지않아 대략 수렴해버렸다.

물론 hyper parameters를 조금씩 만지다가 너무 정책이 빠르게 변하는것같아서 lr를 줄였더니 나름 잘 학습이 되었다.

~~~
import tensorflow as tf
from keras import layers
from keras import backend as K
from keras.models import Model
from keras import optimizers
import numpy as np
~~~

~~~
class Critic:
    def __init__(self,input_dim,output_dim,tau,gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        
        self.model = self.__make_model()
        self.__build_loss_function()
        self.q_gradient = K.function(inputs = self.model.input,\
                                    outputs = tf.gradients(self.model.output,self.model.input[1]))
        
    def __make_model(self):
        state_input_layer = layers.Input(shape=(self.input_dim,))
        action_input_layer = layers.Input(shape=(self.output_dim,))
        
        x = layers.BatchNormalization()(state_input_layer)
        x = layers.Dense(32,activation = 'selu')(x)
        x = layers.concatenate([x,action_input_layer])
        x = layers.Dense(32,activation = 'selu')(x)
        x = layers.Dense(1,activation = 'linear')(x)
        
        model = Model(inputs = [state_input_layer,action_input_layer], outputs = x)
        return model        
    
    def __build_loss_function(self):
        critic_output = self.model.output
        reward_placeholder = K.placeholder(shape = (None,self.output_dim),\
                                          name = 'reward')
        
        critic_loss = K.mean(K.square(reward_placeholder - critic_output)) 
        
        critic_optimizer = optimizers.Adam(lr = 0.0005)
        critic_updates = critic_optimizer.get_updates(params = self.model.trainable_weights,\
                                                     loss = critic_loss)
        self.update_function = K.function(inputs = [self.model.input[0],\
                                                    self.model.input[1],\
                                                     reward_placeholder],\
                                           outputs = [], updates = critic_updates)
                                           
    def train(self,state,action,reward):
        self.update_function([state,action,reward])
              
    def get_gradient(self,state,action):
        return self.q_gradient([state,action])
        
    def soft_update(self,target):
        weights = np.array(self.model.get_weights())
        target_weights = np.array(target.get_weights())
        target_weights = self.tau * weights + (1 - self.tau) * target_weights
        target.set_weights(target_weights)
        return target
        
class Actor:
    def __init__(self,input_dim,output_dim, tau,gamma):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        self.model = self.__make_model()

        self.__make_loss_function()
                
    def __make_model(self):
        input_layer = layers.Input(shape=(self.input_dim,))
        
        x = layers.GaussianNoise(1.0)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32,activation = 'selu')(x)
        x = layers.Dense(32,activation = 'selu')(x)
        x = layers.Dense(32,activation = 'selu')(x)
        x = layers.Dense(self.output_dim,activation = 'tanh')(x)
        
        model = Model(inputs = input_layer, outputs = x)
        return model
    
    def __make_loss_function(self):   
        q_gradient = K.placeholder(shape = (None,self.output_dim),\
                                  name = 'q_gradient_placeholder')
        
        loss =  tf.gradients(self.model.output,self.model.trainable_weights, -q_gradient) #grad_ys임 값 고정위해
        optimizer = optimizers.Adam(lr=0.00005)
        updates = optimizer.get_updates(loss = loss, params = self.model.trainable_weights)
        
        self.update_function = K.function(inputs = [self.model.input,q_gradient],\
                                          outputs = [],\
                                          updates = updates)

    def get_action(self,state):
        return self.model.predict(state)
        
    def train(self,state,grads):
        self.update_function([state,grads])
    
    def soft_update(self,target):
        weights = np.array(self.model.get_weights())
        target_weights = np.array(target.get_weights())
        target_weights = self.tau * weights + (1 - self.tau) * target_weights
        target.set_weights(target_weights)
        return target
~~~

~~~
import random
from collections import namedtuple, deque 
class ReplayBuffer():
    def __init__(self, maxlen=20000, batch_size=32):
        self.memory = deque()
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
    #원래 그냥 sample로 뽑으시던데 replace가 안돼서 나는 pop으로그냥 뽑았다.
        random.shuffle(self.memory)
        return [self.memory.popleft() for x in range(self.batch_size)]
    
    def __len__(self):
        return len(self.memory)
~~~

~~~

#한 epoch당 reward를 normalize하기 위해
#이렇게 했다.
def _compute_discounted_R(R, discount_rate=.999):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    #for t in reversed(range(len(R))):
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
    discounted_r -= discounted_r.mean() 
    discounted_r /= discounted_r.std()

    return discounted_r

def compute_discounted_R(record,discounted_rate = 0.999):
    reward_list = [x[2] for x in record]
    reward_list = _compute_discounted_R(reward_list)
    for i in range(len(record)):
        record[i][2] = reward_list[i]
    return record
~~~

~~~
import gym
env = gym.make("MountainCarContinuous-v0")
~~~

~~~
class Agent:
    def __init__(self,input_dim,output_dim, tau = 0.001, gamma =0.999):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tau = tau
        self.gamma = gamma
        
        self.main_critic = Critic(input_dim,output_dim,tau,gamma)
        self.target_critic = Critic(input_dim,output_dim,tau,gamma)
    
        self.main_actor = Actor(input_dim,output_dim,tau,gamma)
        self.target_actor = Actor(input_dim,output_dim,tau,gamma)
    
        self.target_critic.model.set_weights(self.main_critic.model.get_weights())
        self.target_actor.model.set_weights(self.main_actor.model.get_weights())
        
        self.memory = ReplayBuffer()

    def get_action(self,state):
        return self.main_actor.get_action(state)

    def train(self):
        while (len(self.memory)) > 100:
            data = self.memory.sample()
            states = np.vstack([e.state for e in data if e is not None])
            actions = np.array([e.action for e in data if e is not None]).astype(np.float32).reshape(-1, self.output_dim)
            rewards = np.array([e.reward for e in data if e is not None]).astype(np.float32).reshape(-1, 1)
            dones = np.array([e.done for e in data if e is not None]).astype(np.uint8).reshape(-1, 1)
            next_states = np.vstack([e.next_state for e in data if e is not None])
            
            actions_next = self.target_actor.model.predict_on_batch(next_states)
            #actions_next = self.target_actor.predict_on_batch(next_states)
            Q_targets_next = self.target_critic.model.predict_on_batch([next_states, actions_next])

            Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)

            self.main_critic.train(states,actions,Q_targets)
            action_gradients = np.reshape(self.main_critic.get_gradient(states,actions), \
                                         (-1, self.output_dim))

            self.main_actor.train(states,action_gradients)

            self.target_actor.model = self.main_actor.soft_update(self.target_actor.model)
            self.target_critic.model = self.main_critic.soft_update(self.target_critic.model)
~~~


~~~
agent = Agent(2,1)
~~~

~~~
for iterate in range(100):
    print('iterate : ',iterate)
    record = []
    done = False
    frame = env.reset()
    ep_reward = 0
    while done != True:
        env.render()
        state = frame.reshape(1,-1)
        state = (state - env.observation_space.low) / \
                (env.observation_space.high - env.observation_space.low)
        action = agent.get_action(state)
        next_frame, reward, done, _ = env.step(action)
        record.append([state,action,reward,next_frame.reshape(1,-1),done])
        if reward < 99:
            reward_t = -1.
        else:
            reward_t = 100.
        ep_reward += reward
        frame = next_frame
        print('state : ', state, ', action :', action, ', reward_t : ',reward_t,', reward : ', reward,', done : ',done)
        
        if done :
            record = compute_discounted_R(record)
            #agent.fit(np.array(S).reshape(-1,2),np.array(A).reshape(-1,1), np.array(R).reshape(-1,1))
            list(map(lambda x : agent.memory.add(x[0],x[1],x[2],x[3],x[4]), record))
            #break
        if len(agent.memory)> 1000:
            print('trained_start')
            agent.train()
            print('trained_well')
    else:
        continue
    break
~~~



참고사이트
https://gist.github.com/ByungSunBae/563a0d554fa4657a5adefb1a9c985626?fbclid=IwAR1IkJqquW3vFHtMWWsd5nItHwVIG9b5jIAfsMZuemVVEBbWrfEv9gu-TI8
https://github.com/jiahengqi/ddpg-keras/blob/master/DDPG.ipynb
https://github.com/vharaymonten/MountainCarWithDDPG/blob/master/MountainCarContinuous-v0%20-%20ActorCritic%20.ipynb

---
layout: post
title:  "논문 주제 및 architecture data structure 제작"
subtitle:   ""
categories: deeplearning
tags: paper
---
Attention based edge removal in wired neural architecture for Image classification 으로 처음은 논문 제목을 대략 정했는데, 해보고싶은 후속 연구가

1. stage마다 concatenation 하지않은 network 만들기.

2. attention(나 highway) 사용한 network 구성

3. label을 uniform하게 넣어 network의 activation이 떨어지는 edge 제거, 랜덤 edge

이라 아직 좀더 봐야겠다. 그리고 NAS 방법론에대한 연구를 좀더 해야할 것 같다.

stage에 대한 rough한 자료구조를 생성중임.


해야할 작업

1. node info 좀더 정리하기(input 갯수, Block filter와 kernelsize등)

2. Node단위 디버깅

###

끝낸 작업

1. 자료구조 rough

2. Block단위 디버깅

3. MNIST 디버깅 


~~~
import networkx as nx
import matplotlib.pyplot as plt
import keras
import itertools
import numpy as np

import tensorflow as tf
from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K

from keras.layers import Concatenate
from keras.models import Model
from keras.layers import Dense, Input, Flatten,Dropout,MaxPooling2D
from keras.engine.base_layer import Layer
from keras.layers.convolutional import Conv2D
from keras.layers import ReLU,BatchNormalization, DepthwiseConv2D,Conv2D
from keras.layers import concatenate

#stage 만들어놓고 그안의 랜덤노드 뽑아서 다음노드연결 만들자.
class DAG:
    def __init__(self,name,node_num = 20, neighbor_num = 4, prob = 0.6):
        def extract_key(v):
            return v[1]
        self.name = name
        pre_graph = nx.generators.random_graphs.connected_watts_strogatz_graph(node_num,neighbor_num,prob)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([x for x in range(node_num)])
        self.graph.add_edges_from(pre_graph.edges)
        print('valid = ', nx.is_directed_acyclic_graph(self.graph))
        
        self.edges = sorted(list(self.graph.edges), key = lambda x : x[1])
        self.sorted_edges = [[k,[x[0] for x in g]] for k, g in itertools.groupby(self.edges, extract_key)]
        
        self.attribute_assigning()
        
    def get(self):
        return self.graph
    
    def attribute_assigning(self):
        self.output_nodes = [x for x in self.graph.nodes if self.graph.out_degree(x) == 0]
        self.input_nodes = [x for x in self.graph.nodes if self.graph.in_degree(x) == 0]
        self.hidden_nodes = [x for x in self.graph.nodes if x not in self.input_nodes and x not in self.output_nodes]
        for x in self.graph:
            self.graph.node[x]['name'] = self.name+"_"+str(x)
        for x in self.output_nodes:
            self.graph.node[x]['type'] = 'output_node'
        for x in self.input_nodes:
            self.graph.node[x]['type'] = 'input_node'
        for x in self.hidden_nodes:
            self.graph.node[x]['type'] = 'hidden_node'

        for x in self.sorted_edges:
            self.graph.node[x[0]]['inputed_by'] = x[1]
            
class Block(Conv2D):
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
    def call(self,inputs):
        #x = K.relu(inputs)
        x = K.conv2d(inputs, self.kernel, strides = self.strides, padding=self.padding)
        #print("block ",x)
        x = K.bias_add(x,self.bias)
        x = BatchNormalization()(x)
        if self.activation is not None:
            return self.activation(x)
        return x

class Stage(keras.Model):
    def __init__(self, name, node_num = 20, neighbor_num = 4, prob = 0.6):
        def extract_key(v):
            return v[1]
        self.name = name
        self.graph = DAG(self.name, node_num, neighbor_num, prob)
        self.nodes = self.graph.get().nodes
        self.nodes_info = []
        for node in self.nodes:
            self.nodes_info.append(self.graph.get().nodes[node])
            
        self.trainable_nodes = list()
        for i in range(len(self.nodes)):
            self.trainable_nodes.append(None) #Node -> Block 크기정해줘야함 info에 넣는식으로
        self.trainable_nodes = np.array(self.trainable_nodes)
        super(Stage, self).__init__()
    def call(self,x, *args):
        for i in self.graph.input_nodes:
            self.trainable_nodes[i] = Node(self.nodes_info[i])(x)
        for i in self.graph.sorted_edges:
            print(i)
            #print(self.trainable_nodes[i[0]].output)
            if self.trainable_nodes[i[0]] == None: 
                if len(self.trainable_nodes[i[1]]) == 1 :
                    self.trainable_nodes[i[0]] = Node(self.nodes_info[i[0]])(self.trainable_nodes[i[1][0]])
                    print('base', (self.trainable_nodes[i[0]]))
                    print('inputed', list(self.trainable_nodes[i[1]]))
                else:
                    self.trainable_nodes[i[0]] = Node(self.nodes_info[i[0]])(list(self.trainable_nodes[i[1]]))
                    print('base', (self.trainable_nodes[i[0]]))
                    print('inputed', list(self.trainable_nodes[i[1]]))
            else:
                self.trainable_nodes[i[0]] = Model(list(self.trainable_nodes[i[1]]),(self.trainable_nodes[i[0]]))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(self.graph.input_nodes)
        print(self.graph.hidden_nodes)
        print(self.graph.output_nodes)
        print(list(self.trainable_nodes[self.graph.output_nodes]))
        result = Node({'type':'real_output', 'inputed_by':self.graph.output_nodes})(list(self.trainable_nodes[self.graph.output_nodes]))
        
        result = ReLU()(result)  # 128
        result = BatchNormalization()(result)
        return result
    def compute_output_shape(self,input_shape):
        return input_shape
class Node(keras.Model):
    #info에서 이 노드로 들어오는 노드 갯수 정해줘야함!!!
    def __init__(self, node):
        super(Node, self).__init__()
        self.node = node
        self.node_type = node['type']
        if self.node_type != 'input_node':
            self.inputs = node['inputed_by']
            self.input_len = len(self.inputs)
        else:
            self.input_len = 1
        
        self.block = Block(filters = 64 , kernel_size = 3) ############filters
    def call(self, inputs):
        #print('inputs', inputs)
        train_layer = LayerWeightCalcul(self.input_len)
        #print('node inputs',inputs)
        x = train_layer(inputs)
        #print('node x ',x)
        #print('x',x)
        x = self.block(x)
        #print('x',x)
        return x
    def compute_output_shape(self,input_shape):
        return input_shape
    
class LayerWeightCalcul(Layer):
    def __init__(self,input_len, **kwargs):
        super(LayerWeightCalcul,self).__init__(**kwargs)
        self.input_len = input_len
    def build(self,input_shape):
        self.ws = []
        self.bs = []
        for i in range(self.input_len):
            self.ws.append(self.add_weight("w",(1,),\
                            initializer = initializers.get("glorot_uniform")))
            self.bs.append(self.add_weight("b",(1,),\
                       initializer = initializers.get("zeros")))
        super().build(input_shape)
    def call(self, *inputs):
        #print('layerweightcalcul inputs ',inputs[0])
        #print("ws[0]",self.ws[0])
        
        ##############여기서 왜 차원확장이 (len(inputs),None, width,height,channels)로 되는지 모르겠음.
        ##############그래서일단 앞부분자르고해보기로함.
        
        result = K.relu((inputs[0] * self.ws[0]) + self.bs[0]) 
        if len(result.shape)>4 :
            result = result[0]
        #print('layerweightcalcul first result ',result)
        for i in range(1, len(inputs)) :
            result += K.relu(self.ws[i] * inputs[i] + self.bs[i])
            if len(result.shape)>4 :
                result = result[0]
        #print('layerweightcalcul result ', result)
        return result
    def compute_output_shape(self,input_shape):
        if len(input_shape) == 4:
            return input_shape
        else:
            return input_shape[0]

img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 12

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x0 = Input(shape=(28,28,1))
x0_ = Conv2D(64,kernel_size = 3,strides=1,activation = 'relu',padding='same')(x0)
x1 = Stage('first')(x0_)
x2 = Dropout(0.25)(x1)
x3 = MaxPooling2D(pool_size=(2,2))(x2)
#print(x3)
x4 = Stage('second')(x3)
x5 = Dropout(0.25)(x4)
x6 = MaxPooling2D(pool_size=(2,2))(x5)
x7 = Stage('third')(x6)
x8 = Flatten()(x7)
x9 = Dense(10,activation = 'softmax')(x8)

model = Model(inputs = x0,outputs = x9)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer = 'adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=5)

~~~

---
layout: post
title:  "드디어 exploring randomly wired network 기본적인 뼈대 완성...."
subtitle:   ""
categories: deeplearning
tags: keras
---

쉽게 생각했다가 너무 오래걸려버렸다..

그래프를 만드는 방법이나 여튼 조금 비효율적이지만 제발 그래프만 다 그려지고, 네트워크 연결만 잘되어라 하면서 만들었다....

손이 부들부들떨리네....


~~~
dag.py

import networkx as nx
import numpy as np
import itertools
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
~~~

~~~
#dataloading
import keras

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
~~~

~~~
#block
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
        
        self.bias = self.add_weight(shape = (self.filters,), initializer = self.bias_initializer,\
                                   regularizer = self.bias_regularizer, constraint = self.bias_constraint,\
                                   name = 'bias')
        
    def call(self,inputs):
        #x = K.relu(inputs)
        
        x = K.conv2d(inputs, self.kernel, strides = self.strides, padding=self.padding)
        #print(x)
        #print("block ",x)
        x = K.bias_add(x,self.bias)
        #x = BatchNormalization()(x)
        if self.activation is not None:
            return self.activation(x)
        return x
~~~

~~~

#new node
class NewNode(Layer):
    def __init__(self,input_len):
        super().__init__()
        self.input_len = input_len
    
    def build(self,input_shape):
        if self.input_len > 1:
            self.ws = [self.add_weight("w",(1,), initializer = \
                                   initializers.get("glorot_uniform")) for x\
                   in range(self.input_len)]
            self.bs = [self.add_weight("b",(1,), initializer = initializers.get("zeros"))\
                  for x in range(self.input_len)]
        super().build(input_shape)
        
    def call(self,x):
        if type(x) == list :
            result = (self.ws[0] * x[0] + self.bs[0])
            for i in range(1, (self.input_len)):
                result += (self.ws[i] * x[i] + self.bs[i])
            x = K.relu(result)
        return x
    
    def compute_output_shape(self,input_shape):
        if type(input_shape) == list:
            return input_shape[0]
        return input_shape
~~~

~~~
class RealNode(Model):
    def __init__(self,input_len,filter_size = 32, kernel_size = 3):
        super().__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.block = Block(self.filter_size, self.kernel_size)
        self.node = NewNode(input_len)
    
    def call(self,inputs):
        #print('asd',inputs)
        x = self.node(inputs)
        x = self.block(x)
        return x
    def compute_output_shape(self,input_shape):
        if type(input_shape) == list:
            return input_shape[0]
        return input_shape
~~~

~~~
#network
graph = dag.DAG('test')

nodes_info = [graph.graph.nodes[node] for node in graph.graph.nodes]

nodes = []
for node in graph.graph.nodes:
    if nodes_info[node]['type'] == 'input_node':
        nodes.append(RealNode(1,32,3))
    else:
        nodes.append(RealNode(len(nodes_info[node]['inputed_by']),32,3))
nodes = np.array(nodes)

x_0 = Input(shape = (28,28,1))
x = Conv2D(32, kernel_size = 3, strides = 1, padding='same')(x_0)
x = ReLU()(x)
x = Dropout(0.25)(x)

for node in graph.graph.nodes:
    if nodes_info[node]['type'] == 'input_node':
        nodes[node] = nodes[node](x)
    else:
        input_layer = (nodes_info[node]['inputed_by'])
        if len(input_layer) > 1:
            nodes[node] = nodes[node](list(nodes[input_layer]))
        else:
            nodes[node] = nodes[node](nodes[input_layer[0]])
            
output = RealNode(len(nodes[graph.output_nodes]),32,3)
output = output(list(nodes[graph.output_nodes]))

x_2 = MaxPooling2D(pool_size = (2,2))(output)
x_2 = Conv2D(128,kernel_size = 3, strides = 1, padding = 'same',activation = 'relu')(x_2)
x_2 = Dropout(0.25)(x_2)

x_3 = Flatten()(x_2)
x_3 = Dense(128,activation = 'relu')(x_3)
x_3 = Dense(10,activation = 'softmax')(x_3)
model = Model(inputs = x_0,outputs = x_3)
#model.summary()
model.compile(loss='categorical_crossentropy', optimizer = 'adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=1)

predict_data = model.predict(x_test)
y_test_argmax = np.argmax(y_test,axis=1)
predict_data_argmax = np.argmax(predict_data,axis=1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test_argmax,predict_data_argmax)

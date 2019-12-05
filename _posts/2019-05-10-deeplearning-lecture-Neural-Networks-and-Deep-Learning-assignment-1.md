---
layout: post
title:  "Neural Networks and Deep Learning assignment-01"
subtitle:   ""
categories: deeplearning
tags: lecture
---

cs230 을 들으면서 syllabus를 보니 Neural Networks and Deep Learning를 꼭 들어야 하는 것 같길래
먼저 이 강의를 끝내려고한다. ( 일주일 간의 수강 제한이 있다 ㅠㅠ) 그리고 내용도 기본기에 충실해서 한번 짚고 넘어가기 좋은 것 같다.


# GRADED FUNCTION: basic_sigmoid

~~~

import math

def basic_sigmoid(x):
    """
    
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / ( 1 + math.exp(-x))
    ### END CODE HERE ###
    
    return s

~~~



# GRADED FUNCTION: sigmoid
~~~
import numpy as np # this means you can access numpy functions by writing np.function() instead of numpy.function()

def sigmoid(x):
    """
    
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    x = np.array(x)
    s = 1 / ( 1 + np.exp(-x))
    ### END CODE HERE ###
    
    return s
~~~


# GRADED FUNCTION: sigmoid_derivative

~~~
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    s = (1 / (1 + np.exp(-np.array(x))))
    ds = s * (1 - s)
    ### END CODE HERE ###
    
    return ds
~~~

~~~
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape(-1,1)
    ### END CODE HERE ###
    
    return v
~~~

# GRADED FUNCTION: normalizeRows
~~~
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.sqrt(np.sum(x * x,axis=1, keepdims= True))
    print(x_norm)
    # Divide x by its norm.
    x = x / x_norm
    ### END CODE HERE ###

    return x
~~~

# GRADED FUNCTION: softmax
~~~
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp,axis=1,keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum

    ### END CODE HERE ###
    
    return s

~~~
# GRADED FUNCTION: L1
~~~
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(yhat-y))
    ### END CODE HERE ###
    
    return loss
~~~

# GRADED FUNCTION: L2
~~~
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum((yhat-y) * (yhat-y))
    ### END CODE HERE ###
    
    return loss
~~~

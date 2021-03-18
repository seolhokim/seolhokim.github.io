---
layout: post
title:  "Activation function 이해하기"
subtitle:   "Activation function 이해하기"
categories: deeplearning
tags: etc
---

## 1. Activation function 이란

input value는 보통 layer들과의 dot을 통한 값이 propagation 되어 output이 된다. 이 과정에서 매트릭스가 사용되게 되는데, 이를 수식으로 나타내면 보통은 이런식으로 표현한다.

$$t = \sum_{k=1}^{} w_k x_k + b_k  $$

그리고 이 값에 function을 씌워 다음값으로 넘어가는데, 이때 이 function을 activation function이라고 한다.

$$ f(t) = activation function $$


 근데 나는 이렇게 표현하는게 정확한 것 같은데 참고만하길 바람

$$ X' =  \{ {x}'_1,{x}'_2, ..., {x}'_m \}       , m : output dim $$

$$ {x}'_i = \sum_{k=1}^{} w_{ik} x_k + b_i ,  \{ {x}'_i \in X' \} $$



## 2. Activation function은 왜 중요한가

그렇다면 위의 구조가 deep해진다면 어떻게 될까?

trainable weight A,B,C,D를 각각 가진 4개의 레이어를 통과한다면 아마 $$ X' = ABCDX^t $$ 의 구조로 전달이 될 것이다.

하지만 이는 $$ X' = T X^t (\exists T) $$ 로 전달 가능한 선형이며, 결론적으로 deep해지는 의미가 없다.(backpropagation 에서도 선형이라면 기울기가 상수라 문제가생김) 그러므로 일반적인 선형적 결합을 넘어 그 layer만의 전달력을 살려야한다. 어떻게 그렇게 할 수 있을까?

바로 activation function에 현재 구한 $$X'$$ 를 넣는 것이다. 그렇다면 수식으로 이렇게 나타낼 수 있다. $$ f(x) = activation function x \in X' $$ 그렇게 되면 layer마다 선형적인 관계를 깰 수있다. 이처럼 비선형적인 관계를 이용해 deep한 구조를 만들 수 있는 기본적인 바탕을 깔게 되었다.

이외에도 binary step function이 존재한다.

## 3. 비선형 Activation function의 종류

### 3.1 Sigmoid ( = logistic)

시그모이드 함수는 S자의 형태로 y값이 (0,1) 사이의 값을 갖는 함수이다.

![시그모이드](/assets/img/sigmoid.png)

$$ \sigma(x) = \frac{1}{1+exp(-x)} $$

$$\sigma'(x)= \sigma(x)(1-\sigma(x))$$

신기한 점은 산술 기하 평균 부등식으로부터

$$ \frac{\sigma(x) + (1-\sigma(x))}{2} >= \sqrt[2]{\sigma(x) (1-\sigma(x))} $$

이 도출되고, $$ 0.25 >= \sigma\(x) >= 0 $$ 이 된다.

이는 편미분을 했을 때, 항상 양의 값을 가지게 됨을 의미하는데, back propagation을 하면 항상 양 또는 음의 값으로 parameter를 update하게 된다. 이는 업데이트하고자 하는 파라미터에 대해 update될 때, 일직선으로 update가 될 수 없는 경우가 있어 zigzag 현상이 생기므로 수렴이 늦어지게 된다.

또한, 그림처럼 $$ -6 <= x <= 6 $$ 에서 gradient가 0으로 수렴해버리므로 saturation되어 back propagation 시 input과 가까운 layer 쪽은 update가 잘 안되는 현상이 일어난다. 

주로 결과값을 내는데 사용된다. 이러한 많은 문제점들이 있지만 항상 절대라는건 없다고 생각한다. 어떨 때는 시그모이드가 좀더 좋은 성능을 내기도 하고(드물지만), tanh나 softmax류가 더 잘 나올 때도 있다. 보통 출력값을 내는데 사용하지만 tanh처럼 attention layer 류 에도 적용 가능하다. 출력을 주로 하는 activation function과 hidden 딴에서 주로 사용하는 activation function을 나눠 적을까 하다가, 결국 갇힌 사고인 것 같아서 따로 분리하지 않았고 마음껏 실험을 통해 좋은 결과를 얻어내길 바란다(벌써 AutoML이 실험의 즐거움을 빼앗아 가고 있지만...)

### 3.2 tanh

![하이퍼볼릭탄젠트](/assets/img/tanh.PNG)

$$ tanh(x) =  \frac{ e^x - e^-^x }{ e^x + e^-^x} $$

$$ tanh'(x) = 1 - tanh^2(x) $$

로, $$ -1 <= tanh(x), tanh'(x) <= 1 $$ 이어서 범위가 좀더 늘어났고 zero centered 이다. sigmoid에서 zigzag랑 zero center는 해결됐지만, saturation은 여전하다. 그래도 위 이유들 때문에 수렴이 빠르다.

### 3.3 softmax

$$ f(x_i) = \frac{ e^y^i } { \sum_{j}^{} e^y^j} \all x_i \subset X $$

그러면 결국 $$ \sum_{i}^{} f(x_i) = 1 $$ 이 되어 $$ f(x_i) $$ 은 $$ x_i $$ 가 scaling 되었고 확률처럼 변환 되었음을 알 수 있다.

t 를 넣어서 temporate으로 사용해 얼마나 soft하게 변환시킬지 변환 할 수 있다. 높은 온도 낮은 온도 해서 자연어 처리할 때 온도를 높이면 다양성이 늘음rms 같은 optimizer를 쓰면 미분이 어렵고 crossentropy로 훈련시켜야함... 정보이론도 다음에 설명할 예쩡.. 


### 3.4 relu

![relu](/assets/img/relu.PNG)

$$ f(x) = max(0, x) $$ 

zigzag 문제도 반절은 해결했지만, zero에서 미분 불가능 한  



# (미완)


 







[참고](https://juxt.pro/blog/posts/neural-maths.html)
[이곳](https://subinium.github.io/introduction-to-activation/) 을 참고하여 만들어졌습니다.

sigmoid 사진 위키피디아
tanh 사진 매틀랩
https://medium.com/@kanchansarkar/relu-not-a-differentiable-function-why-used-in-gradient-based-optimization-7fef3a4cecec

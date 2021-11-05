---
layout: post
title:  "Stand-Alone Self-Attention in Vision Models 논문 리뷰"
toc: true
categories: 
  - Deep Learning
tags: [Deep Learning]
author:
  - Seolho Kim
math: true
---

## Abstract

convolutions은 modern computer vision systems block의 fundametal이다. 하지만 최근 연구는 convolutions를 넘어가는 걸 논의한다. long-range dependencies때문에.
이러한 노력들은 convolutional model의 content-based interactions를 증가시키는데 집중되고 있다. self-attention이나 non-local means 같은 예가 있다. 여러 vision tasks 에대한 이득을 얻기위해. 여기서 자연스러운 question은 attention이 attention이 독립적으로 쓰일 수 있느냐이다.
여기서 self-attention이 독립적으로 쓰여도 효과적임을 보임. resnet이나 이런것에서 적은 파라미터 양을 보였음.

## Introduction

Digital image processing은 recognition에서부터 발생, handcrafted linear filters는 convolutionally 하게 pixelated imagery 에 적용 되었고, 다양한 applications 을 낳음. 이러한 성공은 biological consideration에서부터 영감을 얻었고, 이러한 구조는 parameter-efficient 했다.

large dataset과 compute resources 출현은 cnn을 많은 computer vision에 적용되게 만듬. deep learning 의 영역은 cnn architecture를 design하는데 많이 이동함. image recognition이나 object detection이나 image segmentation에서. convolution의 translation equivariance property(위치 기억성 정도라고 보면 편하실듯)는 그걸 block화하는 것에 strong motivation을 줬다. 하지만, 여기서 문제가 long range interactions이다. large receptive field로 인한 poor scaling properties 때문에 발생한다고 하는데, 보통 pooling으로 object scale variability 를 해결하는데 receptive field가 크면 pooling으로 해결 못할 수 있겠다하는 걸 느꼈다. 또한 convolution layer층 간의 interactions이 되기 어려움.

long range interaction의 문제는 attention으로 해결해 왔다. 이는 computer vision model에서도 performance를 보였는데, channel-based attention mechanism(기본적인 마지막에 element-wise하는 attention. SEnetworks 관한 글 올림)이 적용되었다.
유사하게, spatially-aware attention mechanism도 object detection 및 classification 하는데 사용됨. convolutional model위에 global attention을 layer 더하는 방법도 사용됨. 이런 global form은 input의 all spatial locations을 엄청 작은 input으로의 사용을 제한시킴.

이러한 work에서 우리는 question을 하게된다. 만약에 content-based interactions이 primary primitive가 될 수 있지않을까 하는. 마지막으로 여기서는 simple local self-attention layer를 작고 큰 inputs 두 곳에 만들었다. 여튼 뒤에는 fully attentional vision model이 흥하고 새로운 지평 열었으면 좋겠다는 얘기다.

## 2 Background

### 2.1 convolutions

Weight matrix 는 $$ W \in \mathbb{R} ^{k \times k \times d_{out} \times d_{in} } $$ 인거는 전에 kernel size 설명하면서 설명했던 내용이고 여기서는 output이 $$ y_{ij} \in \mathbb{R}^{d_{out}} $$ 을 강조하며, input value를 spatially summing 하는 depthwise conv를 강조한다. 이러한 노력들이 cnn이 deployment되도록했다. 

### 2.2 self-attention

이전에 attention에 대해 많이 연구를 했었는데, 볼 때마다 어려운 것 같다. 돌아가서 보면, attention은 encoder-decoder 할때 소개되었고, 가변길이 information의 content-based summarization를 할때 썼다. 이 attention의 능력은 foucs on important region인데, neural transduction model에서 중요한 요소가 되었다. attention을 representation learning에 중요한 메카니즘으로 사용하는 것은 그 후 많이 adopted되었고, recurrence를 완전히 대체함. self-attention은 single context에서 나옴 (=query key value 모두 같은 same context에서 나옴.) 이러한 self-attention의 long-distance interaction과 parallelizability ability로 인해, 다양한 task에서 sota를 이룸.

self attention으로 강화된 convolution model의 새 theme은 여러 vision tassk에서 많은 이득을 창출함. non-local means denoising 기법에서도 self attnetion이 쓰인다는데 참조한 논문 [52]를 찾아봤는데도 image processing은 약해서 이해하기 어려웠다. ㅠㅠ근데 video action recognition task에서sota를 보이고 많이 쓰이는 것 같다. 거기서는 conv layer를 다없애고 local self-attention만을 이용해 network를 구성했다고 한다. 다른 concurrent work도 비슷한 양상인데, new content-based layer를 모델에 사용하는 것이다. 이러한 접근들은 self-attention을 사용하는 것과 비슷한 양상이다.

여기서는 stand-alone self-attention layer를 spatial convolution을 대체하고 온전한 attentional 모델을 만들수 있다는 걸 제시한다. attention layer는 단순히 이전 works들을 재활용하고, 다음 novel한 모델을 만드는 future work를 제시하기만 한다는 말을한당..

픽셀 $$ x_{ij} \in \mathbb{R}^{d_{in}} $$ 이 주어졌을 때, local region of pixels in position $$ ab \in \mathcal{N} _ k (i,j) $$ 를
추출하는데, spatial extent k 는 $$ x_{ij} $$ 를 주변으로 있다. 이걸 memory block이라고 하고, 이러한 local attention이 이전 prior work와 다르다. vision 영역에서 성행하던 attention이랑.(all-to-all한global attention만 사용했음) global attention은 significant한 downsampling 이후에만 사용가능하다.(비싸니까.)

single-headed attention은 Figure 3번(query key value 계산 그림화함)과 같고 수식은

$$ y_{ij} = \sum_{a,b \in \mathcal{N}_k(i,j)} softmax_{ab} (q_{ij}^T k_{ab}) v_{ab} $$

보통 normalization도 하던데, 여기 수식은 표현을 안해줬다. queries $$ q_ij = W_Q x_{ij} $$ 고, keys $$ k_{ab} = W_K x_{ab} $$ values $$ v_{ab} = W_V x_{ab} $$ 는 input pixel in position $$ ij $$ 의 linear transformation이다. $$softmax_{ab} $$는 ij와 $$W_Q, W_K, W_V \in \mathbb{R} ^{d_{out} \times d_{in} }$$ 에 의해 계산된 logits이다. local self attention은 주위 neighborhoods 의 spatial information을 aggregate 하는 건 convolutions과 같지만, aggregation이 value vector와 weights의 convex한 combination이다. 이러한 연산은 모든 pixel $$ ij$$ 에 대해 반복되며, 실제로 multiple attention heads가 input의 distinct한 representation을 배운다. 그러므로 input feature를 N 그룹으로 나눈다는데 보통 channel 단위로 grouping 할 것 같다. 그러면 $$ x^n_{ij} \in \mathbb{R}^{d_{in} /N}, W^n_Q, W^n_Q,W^n_V \in \mathbb{R} ^{d_{out}/N \times d_{in} / N } $$ 가되고, 다음에 concat 되어 output의 모양은 변하지않는다.

현재 틀에 맞춰져서, positional information은 attention에 encoded되지 않았다. 이것은 permutation equivariant를 만들고, vision task의 표현도를 제한시킴! absolute position에 대해 Sinusoidal embeddings을 하는 방식도 사용되었으나, 최근 연구는 relative positional embedding이 낫다고 결과가 나오고 있따. 반면에 2d relative position embedding이 사용된다. relative attention은 ij와 $$ab \in \mathcal{N}_ k(i,j) $$ 의 relative distance를 구하는 것으로 부터 시작한다. dimension에 따라 상대적 거리는 계산되므로, row offset $$ a - i $$ 과 column offset $$ b- j $$ 가 구해진다. Figure4 보면 이해가 쉬움. row and columns offsets은 embedding $$r_{a-i} $$와 $$r_{b-j} $$ 는 각각 dimension이 1/2된다. (당연히 (row,column)에서 row씩 column씩으로 분할했으니까) 이 spatial-relatvie attention은
 
 $$ y \sum_{a,b \in \mathcal{N}_ k(i,j)} softmax_{ab}(q^T_{ij}k_{ab} + q^T_{ij}r_{a-i,b-j})v_{ab} $$

그냥 qkv에 relative distance를 하나 더 구해준 것과 같다

그러므로, query와 $$ \mathcal{N}_ k (i,j) $$의 logit measuring the similarity 는 content와 relative distance의 결합으로 만들어짐.
position information을 넣음으로써, self attention은 translation equivariance를 가지게된다. 

attention의 parameter 개수는 spatial extent에 독립적이나 convolution은 spatial extent에 대해 2배로 증가한다. 또한 computational cost of attention도 적게 증가한다 convolution보다! $$ d_{in} = d_{out} = 128 $$ 에서 computational cost가 convolution은 k=3일때랑 attention layer k = 19일때랑 같다.

## 3 Fully Attentional Vision Models

local attention layer를 기반으로, question은 어떻게 construct하냐인데 두 스텝으로 이뤄냈다.

### 3.1 Replacing Spatial convolutions

Spatial convolution은 spatial extent 가 1보다 큰 convolution인데 1 x 1 convolution은 제외했다. 왜냐면 얘네는 standard fully connected layer로 여겨진다. 이 work는 creating fully attentional vision model을 탐구하는데, 이미 있는 convolutional architecture를 가져다가 spatial convolution을 attention layer로 교체하고, pooling이나 stride는 spatial downsampling이 필요할때 나온다.
이 작업은 resnet 류의 architecture에 적용해봤다. resnet의 core building block은 bottleneck block인데, 1x1-> 3x3-> 1x1 로 구성됨.
input block과 last convolution output 에 residual connection가 붙는다. bottleneck block은 resnet안에서 여러 번 등장하는데, bottleneck block의 output은 다음 bottleneck block의 input이된다. 3x3 spatial convolution을 self attention layer로 swap했다. 다른 모든 structure는,(layer 갯수나 spatial downsampling) 모두 보존됐다. 이 변환 전략은 단순하지만, suboptimal전략으로 쓸만하다. attention을 core component로 crafting architecture하는 것은 (architecture search 같은 것처럼) 더 좋은 결과를 나오게 할것이다.

### 3.2 Replacing the convolutional stem

CNN의 시작 layer를 가끔 stem이라고 부르는데, 이게 edge같은 local features를 learning하는데 critical role를 한다. (이걸 이 후의 layer에서 전체적 object를 identify 하는데 사용한다.) input image가 크기때문에 stem은 가끔 core block과는 다른데, downsampling 같이 lightweight하는 operation을 한다. 예를 들면, resnet에서 stem은 7x7 convolution(stride 2) 다음에 3x3 maxpooling(stride 2)를 한다.

stem layer에서는 content는 RGB pixels들로 구성되어있다.(개별적으로는 uninformative하지만 spatially correlated 된). 이 요소들은 learning을 useful feature로 만드는데 이건 attention layer로하면 underperform했다고한다.

distance based weight parametrization of convolutions은 그들이 edge dectectors나 다른 local features를 배우기 쉽게 한다. convolution과 self-attention의 연결을하면서 계산은 크게 늘리지 않기 위해, point-wise conv를 넣었다. 

stem은 공간적인 value features를 가지고 있는 attention layer로 이루어져있고 그뒤엔 maxpooling이 있다. 간단하게, attention receptive field는 maxpooling window와 어울린다.






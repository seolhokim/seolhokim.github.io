---
layout: post
title:  "Attention 정리 1"
subtitle:   "Attention"
categories: deeplearning
tags: attention
---

안녕하세요. 이번에 SNS(인스타그램)의 허수계정을 판별해내는 프로세스 및 딥러닝 모델을 제작하면서, 인스타그램내의 글들을 분석하기 위해 character-level로 접근을 했고, 좀 더 효과적으로 사용하기 위해 attention에 대해 좀더 연구를 했습니다.(자세한 개발 내용은 Recent에 올릴예정입니다.) 

포스트 내용은 지극히 주관적이고 연구한 내용을 바탕으로 적은 것이니, 틀리거나 다른 견해가 있다면 상처안받게 잘 남겨주시면 감사하겠습니다 ^^

모든 내용은 Keras와 백엔드로 tensorflow를 중심으로 설명을 합니다.

## Attention-mechanism이란

학습 과정에서의 Attention 이란 raw 한 데이터 내에서 필요한 정보를 선택하는 것을 배우는 mechanism이라고 이해했습니다.

attention은 많은 형태로 구현가능 합니다.

## Layer 별 Attention 구현 형태

### 1) simple attention

가장 attention을 쉽게 이해할 수 있는 layer로, 매트릭스 내의 특정 파라미터의 값을 증폭 혹은 감소시키도록 하는 layer입니다. 그러면 일반 layer와 무엇이 다른지 궁금할 수 있는데, attention nn은 trainable한 matrix를 생성해, input matrix의 dot(+scaling)을 통해  어느 부분을 주목할지 만들어 input matrix와의 inner product를 통해 input matrix의 파라미터에 가중치를 곱하는 형태로 이루어지기 때문에 일반 layer와는 차이가 있습니다. 

[구현 깃](https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d#file-attention-py)

의문점 

1. 위 깃 처럼 layer 구현하면 맨날 보는게 tanh 뒤에 다시 exponatial을 다시 해걸어주던데 scaling의 문제인지, back propagation 쉽게하려는 건지 논문에서도 식은 저렇게 쓰니까 구현은 이렇게하는데 뭐가 더 장점인지는 모르겠음.

### 2) cnn attention

일반 layer에 attention을 사용할 수 있는 것처럼 convolutional layer 에서도 비슷한 형태로 구현할 수 있는데, 가장 간단하게 2dimensional convolutional layer를 보겠습니다.  [캐글 커널](https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age) 를보면 pre-trained 된 vgg-16 모델뒤에 layer를 새로 붙이는 transfer learning을 시도하는데, input layer (inner product) attention layer 형태로 만들기 위해 두가지 branch로 나눠집니다.

  1. vgg-16모델의 output을 normalize 한 branch 

  2. vgg-16 모델의 output을 normalize하고, 다시 convolutional layer층에 넣은 뒤, 마지막에서 activation function을 sigmoid로, 나온 값들을 선형변환해 simple attention 에 있었던 tanh 효과를 내서 1번 branch와 inner product를 진행함. (locallyconnected2d를 사용했는데 데이터의 특성때문에 사용한걸로 보인다.)

convolution layer의 filter 갯수를 주목할 필요가 있는데, 결국 하나로 모아준뒤 복제해 곱해주게 된다. 하지만 나는 feature map 각각의 attention을 따로 적용하는 것도 좋은 아이디어가 될 것으로 보인다.

### 3) rnn attention

[캐글 커널](https://www.kaggle.com/takuok/bidirectional-lstm-and-attention-lb-0-043)

call 부분을 보면, input_value를 (-1, feature_dims)으로 reshape하고(input_data를 일렬로 쫙핌), trainable한 matrix W를 통해 dot을 한다. 이를통해 단어들의 가중치 합이 모두 구해짐. 그러면 역시 tanh와 exp를 씌운뒤 softmax로 스케일링을 해 input_value에 다시 inner product를 진행한다.

### 4) query key value의 attention

사실 이 주제가 글을 쓴 목적이며 query key value가 결합된 attention으로 2편으로 돌아오겠다.





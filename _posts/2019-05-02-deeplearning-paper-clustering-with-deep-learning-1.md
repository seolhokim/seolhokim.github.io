---
layout: post
title:  "Clustering with Deep Learning Taxonomy and New Methods 1편 논문 리뷰"
subtitle:   "Clustering with Deep Learning Taxonomy and New Methods"
categories: deeplearning
tags: paper
---

## Abstract

deep neural networks에 기반한 clustering methods 는 proven 됨.

(neural net의 high representational power때문에!)

그래서 이 논문은 systematic taxonomy of clustering methods using neural net을 보여줌



texonomy 는 한계가 있는 previous methods 들을 선택적으로 재결합 및 제거함으로써
taxonomy가 systematically create new clustering methods 하는 것을 보였다.

## 1. Introduction

clustering의 목적은 similar data를 group으로 분류하는 것이다.
(data analysis 나 data visualization을 위해선 필수적!)

current clustering methods는 input data에 의존적인데, 다른 데이터셋은 다른 측정방법과 다른 분류 방법을 가져야한다. 
그러므로 차원축소나 표현학습 (representation learning) 는 광범위하게 사용된다.
결과적으로, input data를 separation이 쉬운 feature space에 mapping 한다.
dnn을 통해 이런 non-linear한 mapping이 가능해졌고 (추가적인 사람 손에 의한 feature engineering 없이)
feature extraction and clustering 은 k means, auto encoder등으로 적용되었었음


이 논문의 메인 contribution인 논문은 분류 체계의 구성이 deep neural net에 의한 표현학습인 논문이다.
여기서 제안된 분류방법은 사람들에게 새 방법을 만드는걸 가능하게 할거다. 기존의 한계가 있는 방법들의
selectively 재결합 혹은 제거를 통해.
분류는 특히 자기가 만든 블록으로부터 자기가 방법을 만들길 원하는 practitioners 들에게 valuable하다. 
여기서 제안하는 분류를 설명하자면, 이 논문에선 분류학의 insights에 base를 둔 case study를 했음.

case study에서, 이 논문에서는※convolutional autoencoder※를 사용함.
클러스터링이 쉽게 되기 위한 데이터의 representation을 ※ 2단계의 트레이닝 과정※을 통해 배우게됨.


- 첫번째 과정은 autoencoder인데 standard mse loss function이용하면됨

- 두번째 과정은 autoencoder는 fine-tuned 되는데 combined loss (auto encoder + clustering-specific loss)


나머지는 이렇게 구성되어있다. 첫번째로 분류(taxonomy) 설명(섹션1)과 그것의 buiding blocks(구조?)(섹션2).

(섹션 3)은 clustering metohds와 analyze 포괄적인 리뷰(여기서나온 분류(taxonomy)와 함께).

섹션 4는 새로운 방법인데 분류(taxonomy)의 systematic way로 부터 얻은 insight에 기반한 방법 제시됨.

섹션 5는 논문에서 나온 방법을 설명하고, 결론이 섹션 6에서 나온다.

## 2. Taxonomy

deep learning 을 이용한 가장 clustering의 sucessful한 방법은 다음의 원칙을 따른다.

표현학습(representation learning using DNN) 그리고 그 표현을 specific clustering method에 넣는것.

// 생각해보면 이미 떠올렸던 방법임. auto encoder류의 encoder를 만들고 encoding 된 feature를 사용해

// 뭔가를 해보려 했었는데 여기서 적절하게 사용한듯

Figure 1에 설명됐듯이, 모든 방법은 다음과같은 파트로 구성되어 있고, 각각은 여러 선택할 수 있는 옵션들이 있다.

- Neural network training procedure은 이렇게 구성되어 있다.:

  - 메인 뉴럴넷 branch 와 사용
    - 메인 뉴럴넷의 구조는 section 2.1에 묘사됨
    - 클러스터링을 위한 deep features의 집합은 section 2.2에 묘사됨
  - 뉴럴넷 로스
    - non-clustering loss 는 2.3
    - clustering loss 는 2.4
    - method to combine the two loss 2.6
        
- Option. network training 이후에 클러스터링을 재사용 하는 방법 2.7
       
        
### 2.1 main neural network branch

dnn based clustering의 대부분은 (main branch of the neural net(non-clustering loss와는 떨어져있다. 세션 2.3보면됨))
inputs을 clustering을 위해 latent representation 으로 transform하는 게 주로 이루어진다. 
다음과 같은 neural network architectures들이 이런 목적으로 사용되었다.

* multi layer perceptron (MLP) : feed forward network로, several layers of neural networks 로 이루어짐. 
    
* convolutonal neural network(CNN) : biology에서 영감을 받아 만들어졌음. regular-grid data에 효과적
    
* Deep Belief Network (DBN) : generative grahpic model로, 여러 latent variables 층으로 만들어짐.
각 sub network의 숨겨진 계층이 다음 sub network의 visual layer 역할을 하도록 여러 shallow한
networks(restricted boltzmann machines 같은)로 이루어짐

// DBN 대략 보긴 했는데 demension reduction 쪽에서. 음.. 원래 모델의 분포를 구하도록 설계됐던 것 같다.

// 정확히는 까먹음.. 이거보고 좀더 봐야할듯

* Generative Adversarial Network(GAN) : 2개의 competing하는 neural net 2개 로 구성, generator와, discriminator.
Generator는 원래 데이터의 분포를 배우고 discriminator는 분간하는 걸 배우는 네트워크.

* Variational Autoencoder(VAE) : Autoencoder architecture에서 평균과 표준편차를 이용해 좀더 유의미하게 차원을
나누게 됨

### 2.2 Set of deep features used for clustering

input data를 좀더 clustering-friendly 한 representation으로 바꾼뒤엔, features 는 한개나 그이상의 layer에
들어가게 된다.

- One layer : low dimensionality로 인한 이득. 

- Several layers : representation이 여러 layer의 outputs 의 조합일때를 말한다. 그러므로 representation 은 좀더 풍부하고 complex 한 semantic rrepresentations 을 허용한다. 그건 separation 돕고 similarity computation에 도움이됨


### 2.3 Non-Clustering loss

non-clustering loss 는 clustering 알고리즘과 독립적임. 그냥 regularization 인듯

- No Non-clustering loss : additional non-clustering loss 가 없으면 단지 clustering loss 에 의해서만 constrained된다.
대게 clustering loss 들은 non-clustering loss의 부재가 안좋은 representation의 위험이 될 수 있고 collapsing cluster
가 될 수도 있다.(드물게)

- Autoencoder reconstruction loss : the autoencoder는 2가지 파트로 구성된다. encoder와 decoder. encoder는 input x를
representation z 로(latent space Z의) 매핑한다. 훈련동안, decoder는 z로부터 x 를 재구성하는데 힘쓴다. useful infor-
mation 이 encoding phase에서 loss 되지 않도록 하면서. clustering methods 중에서 트레이닝이 완료되면 decoder 파트는
더이상 쓰지않고, encoder만 남아서 latent space Z에 매핑하도록 쓰인다. 이 과정을 적용하면 autoencoder는 성공적으로
input의 dimensionality와는 다른 output의 dimensionality를 가지는 useful representation을 배운다
주로 reconstruction loss 는 distance measure $$ d_{AE}(x_i, f(x_i)) $$ 를 쓴다. x는 X에 속하고 f(x)는 autoencoder
의 reconstruction. 보통 mean squared error를 쓰게 되는데 

$$ L = d_{AE}(x_i, f(x_i)) = \sum_{i}^{}||x_i - f(x_i)||^2$$
로 나타낸다.

이 loss function은 학습된 representation을 보존하는걸 보증한다. 왜냐면 reconstruction이 가능하니까!

- Self-Augmentation Loss : 원형의 representation과 그 augmentations를 함께 넣는 loss이다.
$$ L = - \frac {1} {N} \sum_{N}^{} s(f(x), f(T(x))) $$ 
x 는 원형, T는 augmentation function 이고, f(x)는 model에 의해 generated 된 representation. s는 similarity의
measure function 이다.(cross entropy 같은. )

// augmentation function를 통해 데이터를 생성한 뒤 사용하는 것 같다. over sampling의 logic 에 다른 T(x)를 만들고
similarity를 구해서 regularazation 역할을 할 수 있나보다.

## 2.4 clustering loss

the second type 의 function은 clustering method를 구체적으롭 보여주고, representation 들의 clustering-friendliness 
한 면을 만드므로 clustering loss function 이라고 부른다. 그 function 들은 다음과 같다.

- No clustering loss : a neural network는 오직 non-clustering loss 만을 가진다고 해도, features은 clustering에
사용될 수 있다. neural network는 이 경우, input의 representation을 변화 시킨다. 이런 transform은 clustering에 도움
이 될 수 있다. 하지만 clustering loss 를 사용하는게 대게 더 좋은 결과를 낸다.

- k-Means loss 
새로운 representation은 k-means friendly 할 것을 장담한다. 즉 데이터 포인트들은 cluster center에
균일하게 distributed 될 것이다. 이런 분포를 얻기 위해서 neural network은 다음의 loss function으로 학습되어야 한다.

$$ L(\theta) = \sum_{i=1}^{N} \sum_{k=1}^{K} s_{ik}||z_i - \mu_k||^2 $$

$$z_i$$ 는 embedding 된 data point이고, $$ \mu _{k} $$ 는 cluster center. $$s_{ik} $$는 boolean variable인데, 
$$z_i $$와 $$ \mu_{k} $$ 를 할당해주기 위해서 존재한다.(xi가 Sk에 속하는지 보여주기 위한 digit representation)
이 loss 를 minimizing 하는 것은 network parameters 관점에서
each data point와 cluster center의 거리를 minimizing 되는 것과 같다. 그럼으로써, k means 를 적용하는 것은 cluster
ing quality를 높여줄 것이다!

- Cluster assignment hardening
data를 cluster에 soft한 assignment하는 것이 필요하다. 예를 들면, t분포는 points와 centroids 사이에 similarity를 
계산하기 위한 kernel로 사용된다. 그러므로 distribution Q는 다음의 공식을 따른다. 

$$ q_{ij} = \frac {(1 + ||z_i - \mu_j||^2 / \gamma)^{- \frac {\gamma + 1}{2}}}
{\sum_{j'}^{}(1+ ||z_i - \mu_{j'}||^2/\gamma)^{- \frac {\gamma+1}{2}}} $$

$$z_i $$ 는 embedding 된 data point 를 나타내고 $$ \mu_j $$ 는 $$j^{th} $$ cluster의 centroid이다.
$$ \gamma $$ 는 constant이고 1 이다. points와 centroids 사이의 normalized 된 similarities는 soft cluster assignment
로 여겨진다. cluster assignment hardening loss 는 그러므로 soft assignment probabilities를 엄격하게 만든다.

##2019-05-02


// $$gamma $$ softmax 처럼 temporature 로 쓰는 것 같음 --> soft clustering 을 위함

그렇게 함으로써, cluster assignment probability distribution인 Q는 실제 target distribution P에 가도록 해준다.

$$ p_{ij} = \frac {q^2_{ij}/ \sum_{i}q_{ij}} { \sum_{j'}(q_{ij'}^2/ \sum_{i}q_{ij'})} $$'
                                                                               
원래 분포에 제곱을 해줌으로써, normalizing하게 되고, 실제 분포 P를 0에서 1사이에 위치시킨다. 
high confidence를 가진 data points 를 강조하고 hidden feature space에 왜곡되지 않도록해서 
cluster purity를 올려주기위함! 
                                                                               
두개의 확률 분포를 만드는방법은 쿨백 레이블러 divergence 가 있음 
                                                                               
//정보이론 공부 좀더 해야지 더알고싶다 ㅠㅠ

$$ L = KL(P||Q) = \sum_{i}^{} \sum_{j} ^{}p_{ij}\log(\frac {p_{ij}}{q_{ij}})$$

이 쿨백 레이블러 함수도 Q와 P의 차이를 minimize해줌
                                                                               
- Balanced assignments loss 

이 loss 는 다른 loss function과 함께 사용되었다. goal은 balanced cluster assignment를 해주기 위함!
식은 다음과 같다.

$$L_{ba} = KL(G||U) $$
U는 uniform distribution, G는 각 cluster에 assignment될 확률로

$$ g_k = P(y = k) = \frac {1}{N} \sum_{i}q_{ik} $$ 이다

L을 최소화 함으로써, 각각의 point가 cerntain cluster로 assigning 될 확률이 균등해지고, 이 성질이 언제나 desired
되는건 아니라는걸 알아야한다.  그러므로, 어떤 좋은 방법이 있으면 이방법을 안써도된다.

- Locality-preserving loss 

이 loss 는 cluster의 locality를 유지하기 위한 loss로, 근처의 data points를 함께 넣는다. 수학적으로는

$$ L_{lp} = \sum_{i}^{} \sum_{j \in N_k(i)}^{} s(x_i,x_j)||z_i - z_j||^2 $$

$$N_k(i)$$는 data point $$x_i$$ 의 nearest neighbors들의 set!

- Group sparsity loss 
representation learning 을 위한 block diagonal similarity matrix를 활용하는
spectral clstering 으로부터 영감을 받았다.  

group sparsity 는 그자체로 feature selection method로 effective 하다. 
hidden units들은 G(number of cluster) group으로 divide 된다. data point $$x_i$$ 가 input으로 들어왔을때,
얻어진 representation 은 $${\phi^g(x_i)}^G_{g=1}$$이 되는데 식은 다음과 같다.

$$ L_{gs} = \sum_{i=1}^{N} \sum_{g=1}^{G} \lambda_g||\phi^g(x_i)|| $$

이고 $$ \{\lambda_g\}^{G} _ {g=1}$$ 은 sparsity group에 대한 weights고 수학적으로

$$ \lambda_g = \lambda \sqrt {n_g} $$

이때 $$n_g$$는 group size고, $$\lambda $$는 상수이다.

- cluster classification loss 
cluster assignments는 cluster update간에 'mock' 이라는 class label을 사용할수있다. additional network branch
에서. 좀더 의미있는 feature extraction을위해.

- Agglomerative clustering loss
agglomerative clustering은 비슷한 2개의 cluster를 뭉친다.until some stopping criterion이 충족될때까지.
agglomerative clustering에 의해 inspired된 neural net loss는 여러 스텝동안 계산된다.
첫째로, cluster update step은 가장 비슷한 serveral pairs of clusters를 합친다. 그 뒤 계속, training은
retrospectively  optimize한다. 이미 merged된 clusters에 대해서.
next cluster update step 다음에, network training은 새롭게 merged 된 cluster pair를
retrospectively optimizing하기 위해 switch 된다.
이런 방법으로, cluster merging과 retrospective latent space adjestments는 계속된다.
이 loss function으로 optimizing 된다면, clustering space는 clustering에 더 적합한 space가 될 것이다.
                                                                               
##2019-05-03

### 2.5 Method to combine the losses

당연하게도 $$ \alpha $$값을 조절해 non-clustering loss 와 clustering loss 를 조절할 것 처럼 보임

이경우 clustering과 non-clustering loss function이 같이 사용되는데 이렇게 결합된다.

$$ L(\theta) = \alpha L_c(\theta) + (1 - \alpha)L_n(\theta) $$

$$L_c(\theta) $$ 는 clustering loss고, $$L_n(\theta)$$는 non-clustering loss로, $$ \alpha \in [0;1] $$이고,
constant value이다.(보통 유저가 주는데 non-clustering loss 는 restriction 용이니까, $$ \alpha $$ 는 0.90~0.99) 이정도 사이 값
이지 않을까 싶다. additional hyperparameter라고 여기서도 말한다. training 중간중간 어떤 스케줄에 의해 $$\alpha $$값이
변할 수 있다하는데, 다음과 같다.

- Pre-training, Fine-tuning
$$\alpha$$ 값은 0으로 고정되고, non-clustering loss에 의해서만 학습된다. 그다음 $$\alpha$$ 값을 1로 잡고, non-clustering
network는 뗀다음 이네트워크를 학습한다. reconstruction loss 에 의한 constraint는 학습하면서 사라질 것이고,
가끔 이런경우는 안좋은 결과 초래!

- Joint training
$$ 0 < \alpha < 1 $$ 이어서 두 loss 의 영향을 받음

- Variable schedule
$$ alpha $$ 는 스케줄에 따라 training 도중에 다양한 값이 되는데, 예를들면, $$\alpha $$ 값이 낮았다가 증가.

### 2.6 Cluster Updates

Clustering methods는 크게 hierarchical 하거나 partitional(centroid-based)한 것 두 가지가 존재함.
Hierarchical clustering은 hierarchical한 clusters나 data points를 build하는것에 aim 함. 반면에 patitional clustering
groups methods는 cluster centers를 만들거나 각각에 data points가 cluster에 할당되기 위해 metric relations을 사용
한다.

clustering을 위한 deep learning의 내용을 보면, 2개의 dominant 한 methods들이 있는데, 
Agglomerative clustering과 k-means고 이전에 대략 설명됨.

k-means를 살펴보면, updating cluster assignments는 두개중 하나의 form을 가지고 있다.

- Jointly updated with the network model 

cluster assignments는 확률적으로 생성되어서 back-propagation을 통해 update됨.

- Alternatingly updated with the network model

cluster assignments는 model update 될 때, 엄격하게 update 된다.(정수로 인듯) 이 때, 여러 시나리오가 가능한데,
2개의 main factor만 봄.

  - number of iterations 
    
    정한 clustering algorithm을 여러번 돌림. 
    
  - frequency of update
    
    얼마나 자주 cluster update를 하는지. Yanget al.(2016b)보면, every P network model update step 마다, one cluster
     updates step이 일어났다.

### 2.7 After Network training

training이 끝나면, clustering result가 생성됐을지라도,  scratch(다른 데이터인듯)에서 한번 re-run 한다.
그 이유는 다음과 같다.

- clustering a similar dataset

the general and the trivial case가, 다른 data(비슷하지만 다른 데이터)에 fit된, learning features representation mapping을 
reuse하는 것이다.

- obtaining better result

특정 경우, learning procedure 보다 training 이후에서 더 좋은 clustering 결과가 나올 수 있다. training 도중에 cluster
update를 끝가지 않했을 때.

##2019-05-06

// 갑자기 떠오른 아이디어.

// 음성이나 게임, 이런거 auto-encoder로 embedding 시킨다음에 state 및 action을 할당시켜 학습시키기.

---
layout: post
title:  "Information thoery 이해하기 1"
subtitle:   "Information theory 이해하기"
categories: deeplearning
tags: etc
---
맨날 정보이론부분만 나오면 헷갈렸는데 좋은 정리 블로그를 찾아서 한번 따라 정리해보려고한다.

information 는 발생확률 $$P(x_i) ,x_i \in X $$ 이 낮은 사건 $$ x_i $$는 큰 정보량을 가지고 있다는 것을 기본으로 한다.

이를 information thoery를 놀람의 정도로 비유할 수 있는데, 그렇다면, 낮은 발생확률을 가진 information은 큰 놀람을 가져오니 큰 정보량을 가졌다고 할 수 있다.

이런 정보량은 세가지의 필요 조건이 있는데,

- 자주 발생하는 사건은 낮은 정보량을 가진다. 발생이 보장된 사건은 그 내용에 상관없이 전혀 정보가 없다는 걸 뜻한다.
- 덜 자주 발생하는 사건은 더 높은 정보량을 가진다.
- 독립사건(independent event)은 추가적인 정보량(additive information)을 가진다. 예컨대 동전을 던져 앞면이 두번 나오는 사건에 대한 정보량은 동전을 던져 앞면이 한번 나오는 정보량의 두 배이다.

첫번째와, 두번째는 방금의 놀람의 비유로 충분히 설명이 되지만 세번째를 만족하기 위해서

log를 사용한다.

이를 종합해서 나타낸 정보량은

### information (정보량)

$$ I = - \log {P(x)} , x \in X $$

로 나타낼 수 있다.

### entropy (엔트로피) 

엔트로피는 이러한 사건에 대한 모든 정보량의 평균으로, 섀넌 엔트로피라고 한다.

엔트로피를 수식으로 나타내면

$$ entropy = E(-\log{P(x)}) $$

로 나타낼 수 있다. 이는 모든 확률이 같을 때, 엔트로피가 가장 높다는 것을 알 수 있다.

확률이 0.5, 0.5인 사건의 결과가 나왔을 때, 확률이 0.1, 0.99인 사건보다 놀람이 크다는 걸 예로 들 수 있다.

### KL-divergence (쿨백 라이블러 발산)

kl-divergence 는 상대적인 엔트로피를 의미한다.

이는 확률 $$ P(x) $$ 와 $$ Q(x) $$ 에 대해, 각각의 엔트로피

$$ kl-divergence = E_{X~p}(-\log{Q(x)}) - (E_{X~p}(-\log{P(x)})) $$

를 계산함으로써,  두 분포의 평균의 차이가 작아질 수록 두 분포가 비슷해진다고 할 수 있다.

이는 $$ D_{KL}(P || Q) = H(P,Q) - H(P) $$ 로 나타낼 수 있다.

### cross entropy (크로스 엔트로피)

위에서 말한 확률 $$P(x)$$ 와 $$Q(x) $$ 에 대해, 

$$P(x)$$가 실제 분포, $$Q(x)$$가 구해낸 분포라고 했을 때,

$$P(x)$$는 우리가 알아내고 싶어하는 분포이기 때문에 $$Q(x)$$의 분포를 변화시켜야 한다.

그러므로, $$ H(P,Q) = -E_{X~p}(\log{Q(x)}) $$ 가 된다.









### 참고 블로그 
http://blog.naver.com/PostView.nhn?blogId=gyrbsdl18&logNo=221013188633&redirect=Dlog&widgetTypeCall=true
https://ratsgo.github.io/statistics/2017/09/22/information/

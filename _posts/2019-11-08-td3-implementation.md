---
layout: post
title:  "Addressing Function Approximation Error in Actor-Critic Method (TD3) 구현"
subtitle:   ""
categories: deeplearning
tags: reinforcementlearning
---

구현 기본 form을 여기서 잡을 것이다.
- critic network는 2개, actor network는 1개.(초기화 수작업하자)
- target network로 각각 2개, 1개. 본 네트워크와 같게 초기화
- replay buffer
- $$ \epsilon $$ ~ $$ \mathcal{N}(0,\sigma) $$ 따르는 $$ \epsilon $$ 생성기
- $$ \epsilon $$ ~ $$ \mathcal{N}(0,\tilde{\sigma}) $$따르는 $$ \epsilon $$ 생성기
- critic normalization
- critic loss
- target d cycle update
- actor loss
- soft update

구현은 https://github.com/seolhokim/The-Easiest-TD3-using-pytorch-in-Pendulum 이곳에 되어있다.

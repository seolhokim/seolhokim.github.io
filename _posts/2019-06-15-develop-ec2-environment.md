---
layout: post
title:  "ec2 jupyter server 설정하기"
subtitle:   ""
categories: develop
tags: aws
---
학생 크레딧으로 $100를 받고, jupyter server를 띄워서 노트북으로 작업하려했다.

이전엔 sudo 권한으로 conda환경을 키지 않아도 서버 내에서 작업하는데 문제가 없었다.

근데 이번에 jupyter 서버를 띄워놓고 노트북으로 접근하려니 permission error가 나서 또 삽질을 했다 ㅠㅠ..

결국 root에 anaconda3 path설정이 없어서 간단히

ubuntu 폴더에서

export PATH=~/anaconda3/bin:$PATH

그니까 export PATH=/home/ubuntu/anaconda3/bin:$PATH 을 했었는데, 갑자기 path 설정이 잘못됐었는지,
모든 명령어가 안먹어서 해결책으로

export PATH=/usr/bin:/bin

를 사용했다.

여튼 위의 방법으로  source activate tensorflow_p36이 먹혀서 맨날 멍청하게 ubuntu계정으로 conda환경 띄웠는데 드디어 root로 띄우게되었다..

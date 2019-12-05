---
layout: post
title:  "aws에서 jupyter notebook 서버 띄우기"
subtitle:   ""
categories: develop
tags: aws
---

공짜로하기위해서 해야할 것 2가지가 있다.

허가를 2개 받아야하는데 

## 1.aws에서 student 인증을 받아야함

하루정도 걸리니까 빨리받아야함..

## 2. 인스턴스 제한늘려줘야함

1원이라도 싼 g3s.xlarge 쓰고싶었는데 g3s는 korea region에서 limit release 요청이 없길래 p2.xlarge 3개 풀어달라요청했더니 1 개 풀어줬다..

## 3. 인스턴스 생성

알아서 잘 생성.. ubuntu deep learning ami로 생성하는걸 추천. 인바운드 tcp 다여는거 추천. 

## 4. path 설정
sudo su 로 루트계정 가준다음, source activate tensorflow_p36 하면되는데 path가 설정안되있으므로
~~~
export PATH=/home/ubuntu/anaconda3/bin:$PATH
~~~
해놓고
~~~
source activate tensorflow_36
~~~
해준다.

## 5. password 생성
tensorflow_p36 conda environment에 가면,

python 실행하고 

~~~
from notebook.auth import passwd
passwd()
~~~
해서 비밀번호를 치고 나온 sha값을 저장해논다.

## 6. jupyter notebook config 설정

이제 jupyter notebook 설정을 해줘야하는데, 

jupyter notebook --generate-config 를한다. overwrite 할거냐하면 그냥 해도된다.

생성된 디렉토리를 알려줄건데, vi로 켜서 shift + g 해서 맨 아랫줄에
~~~
c = get.config()
c.NotebookApp.password = u'sha1:@@@@@@' #ex)  u'sha1:a7f7b4ac7e23:1234e091d32cdsb9e5f0ae6912345dbabcd57342'
c.NotebookApp.ip = '@@@' #ex) '172.31.88.13' 프라이빗ip를 입력해줘야함
c.NotebookApp.open_browser = False
c.NotebookApp.notebook_dir = u'/home/ubuntu' #절대경로로 쓰는게안전
~~~

저장한뒤


## 7. 서버 실행

~~~
jupyter-notebook --allow-root
~~~

치면 이제 퍼블릭 ip로 접속가능.

conda 환경에 맞는 script 만든다음에

## 8. 확인

~~~
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
~~~
해서 device 찾았을 때, gpu까지 잘잡힌다면 성공

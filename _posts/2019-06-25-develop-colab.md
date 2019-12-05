---
layout: post
title:  "colaboratory 에서 local drive를 간단하게 사용하는 법"
subtitle:   ""
categories: develop
tags: colaboratory
---

google drive를 일단 설치해서 지정 경로로 local folder랑 연동을한다.

그곳에 파일을 옮긴뒤

colab에서 노트를 하나 만듬

~~~
from google.colab import drive
drive.mount("gdrive")
~~~

후 authorization code 를 입력하면 mounted at gdrive라고 나오는데 그러면 마운트된 것임.

~~~
import glob
glob.glob("./gdrive/My Drive/*")
~~~
해서 확인해보길

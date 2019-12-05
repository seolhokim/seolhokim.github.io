---
layout: post
title:  "windows 10 pytorch 설치 및 troubleshooting"
subtitle:   ""
categories: deeplearning
tags: pytorch
---

windows 10 에서 pytorch를 사용하려면 third-party의 package를 다운로드받아야한다.
~~~
conda create (이름) python=3.6
conda activate (이름)
~~~
으로 환경을 만들고, 환경을 activate 한다.
~~~
conda install -c peterjc123 pytorch
~~~
를 통해 pytorch를 받았는데, 

~~~

    from torch._C import *
ImportError: DLL load failed: 지정된 모듈을 찾을 수 없습니다.
~~~
이러한 에러가 떴다.

~~~
pip install intel-openmp
~~~

을 통해 해결할수 있다고 했는데 안됐고,

GPU가 없는 컴퓨터라 그런거라고 하길래

~~~
conda install -c peterjc123 pytorch-cpu
~~~

로 설치했더니 잘 설치되었다.
~~~
(unity) D:\>python
Python 3.6.7 |Anaconda, Inc.| (default, Dec 10 2018, 20:35:02) [MSC v.1915 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.Tensor([1,2,3])

 1
 2
 3
[torch.FloatTensor of size 3]
~~~
하지만 이 torch는 버전이 낮아서 현재 1.0.0대의 버전의 문법과 조금 안맞는 부분이 있었다.

그래서 다시 

~~~
pip install --upgrade pip

conda update pytorch
~~~
를 한뒤, 

~~~
python

>>> import torch
>>> torch.__version__
'1.0.1'
~~~

---
layout: post
title:  "colaboratory 에서 local drive module import"
subtitle:   ""
categories: develop
tags: colaboratory
---

~~~
import sys
sys.path.insert(0,"gdrive/My Drive/")
~~~

한뒤 import 하자~!

refresh는 
~~~
from importlib import reload
reload(module_name)
~~~

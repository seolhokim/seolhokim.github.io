---
layout: post
title:  "Jupyter notebook에서 Flask돌릴 때"
subtitle:   ""
categories: develop
tags: python
---
커맨드에서는 잘 띄워지는데 jupyternotebook에선 계속 에러(UnsupportedOperation: not writable)가 나서 보니깐


You cannot run a permanent server inside a notebook kernel. Or more precisely, it doesn't make sense. The kernel is there to execute code snippets from a notebook or other client and return the output and results, not to permanently run a server application that listens on a different port. If you want to run a permanent server that listens on its own port, then start it from the command line. The notebook kernel doesn't add any value for that use case.

라고 한다.


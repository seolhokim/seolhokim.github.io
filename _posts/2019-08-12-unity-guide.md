---
layout: post
title:  "유니티를 사용한 머신러닝 환경만들기"
subtitle:   ""
categories: develop
tags: unity
---

나도 이제 시작했지만, unity를 통해서 강화학습 관심있는 사람들은 이정도도 도움될 것 같아서 남긴다.

unity를 사용해 강화학습 환경을 만들 때 다들 튜토리얼에서, 거의 모두가 이미 코딩된 ppo를 사용하는 걸 보여준다.

18년도 환경기준( tutorial들이 거의 이환경에서 실행된다)

ml-agents\mlagents\trainers 에 ppo가 있다.

하지만 ppo를 쓰고싶은게 아니기때문에, 일단

Scene을 Build한다음, 그 폴더에 jupyter notebook 파일을 하나만든다.

~~~

env_name = './Unity Environment.exe' # Name of the Unity environment binary to launch
train_mode = True  # Whether to run the environment in training or inference mode

import matplotlib.pyplot as plt
import numpy as np
import sys

from mlagents.envs import UnityEnvironment


print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

env = UnityEnvironment(file_name=env_name)

default_brain = env.brain_names[0] #3DBallSeol
brain = env.brains[default_brain]


for episode in range(10):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        action_size = brain.vector_action_space_size
        if brain.vector_action_space_type == 'continuous':
            env_info = env.step(np.random.randn(len(env_info.agents),
                                                action_size[0]))[default_brain]
        else:
            action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
            env_info = env.step(action)[default_brain]
       
        episode_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))

env.close()
~~~

여기서 주의해야할 점은 academy에 brain들을 다지우고 내꺼만 남긴상태에서 안하니까 계속 오류가 났다..

state와 action 받아오는거 state space size 잘 맞춰주고 actor에도 brain을 내꺼로 등록한상태에서 build해야한다.

이제 env.step부분에서 actor의 action space size에 맞춰서 넣어주면된다.

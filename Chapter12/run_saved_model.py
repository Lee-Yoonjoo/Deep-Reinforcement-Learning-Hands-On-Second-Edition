import time

import gym
import numpy as np
import torch
from torch import nn
from lib.models import LunarA2c

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    net = LunarA2c(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load('saves/a2c-lunar-lander-a2c/best_+24.935_46911.dat', map_location=torch.device('cpu')))

    obs = env.reset()
    steps = 0
    sm = nn.Softmax(dim=1)

    start_time = time.time()

    while True:
        obs_v = torch.DoubleTensor(obs.reshape(1,8))
        act_probs_v = sm(net(obs_v)[0])
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        steps += 1
        env.render()
        if is_done:
            print(f'game finished after {steps} steps in {time.time() - start_time:.2f} seconds')
            break
        obs = next_obs



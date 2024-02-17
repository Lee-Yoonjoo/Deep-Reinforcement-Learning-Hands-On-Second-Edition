import gym
import numpy as np
import torch
from torch import nn

from lib.dqn_extra import DuelingDQN

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    net = DuelingDQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load('models'))

    obs = env.reset()
    steps = 0
    sm = nn.Softmax(dim=1)
    print(env.observation_space.shape)

    while True:
        obs_v = torch.DoubleTensor([obs])
        print(obs_v)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        #action = env.action_space.sample()
        next_obs, reward, is_done, _ = env.step(action)
        steps += 1
        env.render()
        if is_done:
            print(f'game finished after {steps}')
            break
        obs = next_obs



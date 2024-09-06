from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from rl.rl_toolkit import get_env


class ReplayMemory:
    """
    DQN replay memory具备的特点：
    1.无重复抽样：每次训练从memory中随机抽取一个无重复的batch
    2.定长先进先出队列：采样固定长度L的deque维护最新生产的L个样本
    3.数据格式适配DQN公式：state->action->reward->next_state, 四元组
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, is_terminal):
        self.buffer.append((state, action, reward, next_state, is_terminal))

    def sample(self, batch_size, return_tensor=False, random=True):
        batch_size = min(len(self.buffer), batch_size)
        if random:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)  # replace=False => 不放回抽样
        else:
            indices = list(range(batch_size))
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, is_terminal = zip(*batch)  # 小trick

        # r = [x for x in rewards if x > 80]
        # if len(r) > 0:
        #     print("batch 中奖励大于80个数：", len(r), r)

        batch_state = np.array(states)
        batch_action = np.array(actions)
        batch_reward = np.array(rewards)
        batch_next_state = np.array(next_states)
        batch_terminal = np.array(is_terminal)

        if return_tensor:
            batch_state = torch.tensor(batch_state, dtype=torch.float)
            batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float).unsqueeze(1)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float)
            batch_terminal = torch.tensor(batch_terminal, dtype=torch.float).unsqueeze(1)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal

    def pop(self, n):
        n = min(n, len(self.buffer))
        for _ in range(n):
            self.buffer.popleft()

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self, state_dim, action_dim, epsilon, learning_rate, gamma, target_network_update_freq):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.eval_net = QNet(state_dim, action_dim)
        self.target_net = QNet(state_dim, action_dim)

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma

        # 模型更新
        self.update_counter = 0
        self.target_network_update_freq = target_network_update_freq
        self.optimizer = optim.Adam(self.eval_net.parameters(), learning_rate)
        self.loss = nn.MSELoss()

    def act(self, state, infer=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if infer or np.random.randn() > self.epsilon:
            with torch.no_grad():
                action_value = self.eval_net.forward(state)  # 得到各个action的得分
                action = torch.max(action_value, 1)[1].data.numpy()  # 找最大的那个action
                action = action[0]  # get the action index
        else:
            action = np.random.randint(0, self.action_dim)  # 探索
        return action

    def update(self, memory: ReplayMemory, batch_size: int):
        # 每更新target_network_update_freq次eval net然后就更新一次target network
        if self.update_counter % self.target_network_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # target直接加载eval的权重
        self.update_counter += 1

        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal = memory.sample(batch_size,
                                                                                                  return_tensor=True,
                                                                                                  random=True)

        q_eval = self.eval_net(batch_state)
        q_eval = q_eval.gather(1, batch_action)  # 得到当前Q(s,a)
        with torch.no_grad():
            q_next = self.target_net(batch_next_state)  # 得到Q(s',a')，下面选max
            q_next = q_next.max(1)[0].unsqueeze(1)
            q_target = batch_reward + (1 - batch_terminal) * self.gamma * q_next  # 公式

        loss = self.loss(q_eval, q_target)  # 差异越小越好
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run():
    # 准备环境Env
    env_name = "LunarLander-v2"
    env_name = "CartPole-v0"
    env, state_dim, action_dim = get_env(env_name, render_mode='human')
    # env, num_states, num_actions = get_env("MountainCar-v0")

    epsilon: float = 0.8
    gamma: float = 0.99
    lr: float = 0.001
    memory_capacity: int = 800
    q_network_iteration: int = 10
    batch_size: int = 32
    episodes: int = 40000
    render: bool = False

    model_parameters = dict(state_dim=state_dim, action_dim=action_dim, epsilon=epsilon,
                            learning_rate=lr, gamma=gamma,
                            target_network_update_freq=q_network_iteration)
    agent = DQN(**model_parameters)

    # 经验池：无重复随机抽样, 先进先出队列
    memory = ReplayMemory(memory_capacity)

    print("The DQN is collecting experience...")
    seed = None
    accum_reward = 0
    accum_length = 0
    log_interval = 20
    max_timesteps = 800
    target_reward = 200

    step_counter = 0
    update_interval = 1

    for episode in range(1, episodes + 1):
        state, _ = env.reset()  # seed=seed
        episode_len, episode_reward = 0, 0.  # 本轮游戏走了多少步, 总共获得的奖励
        for _ in range(max_timesteps):  # 每轮游戏最多操作max_timesteps步,即单轮游戏序列长的上限(防止原地转圈或卡死等bug)
            step_counter += 1

            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            memory.store(state, action, reward, next_state, done)  # 记录当前这组数据

            episode_len += 1
            episode_reward += reward
            # if reward > 80:
            #     print("!!!!!!  reward {} !!!!!!".format(reward))

            if render:
                env.render()
            if done or episode_reward < -400:
                # print("episode {} done at step {} with reward {}".format(episode, episode_len, round(reward, 3)))
                break

            state = next_state

            if len(memory) >= batch_size * 4 and step_counter % update_interval == 0:  # 攒够数据一起学
                agent.update(memory, batch_size)
                # memory.pop(batch_size)  # 强制弹出早期的样本

        if episode % 20 == 0:
            print("epsilon: {} episode: {}".format(agent.epsilon, episode))
            agent.epsilon = max(0.15, agent.epsilon * 0.98)

        if episode % 20 == 0:
            reward_sum = 0
            actions = []
            rewards = []
            state, _ = env.reset()
            for _ in range(1200):
                action = agent.act(state, infer=True)
                next_state, reward, done, truncated, info = env.step(action)

                actions.append(action)
                rewards.append(round(reward, 3))
                reward_sum += reward
                env.render()
                if done:
                    break
                state = next_state
            print("*** Test reward", reward_sum)
            print("*** actions", actions)
            print("*** rewards", rewards, '\n')
            # 当最新的累积奖励高于预设阈值时,则停止训练
            if reward_sum > 600:
                print("训练目标已达成：平均奖励超过阈值")
                torch.save(agent.eval_net.state_dict(), './DQN_{}.pth'.format(env_name))
                break

        # 累积本轮序列长,总奖励
        accum_length += episode_len
        accum_reward += episode_reward

        # logging日志：会重置累积奖励和序列长度
        if episode % log_interval == 0:
            avg_length = int(accum_length / log_interval)  # 每一轮游戏平均玩多少步
            avg_reward = int((accum_reward / log_interval))  # (到目前为止)每一轮游戏的奖励均值

            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
            accum_reward = 0
            accum_length = 0



def eval():
    # 准备环境Env
    env_name = "CartPole-v0"
    env, state_dim, action_dim = get_env(env_name, render_mode='human')
    # env, num_states, num_actions = get_env("MountainCar-v0")

    epsilon: float = 0.8
    gamma: float = 0.99
    lr: float = 0.001
    q_network_iteration: int = 10
    model_parameters = dict(state_dim=state_dim, action_dim=action_dim, epsilon=epsilon,
                            learning_rate=lr, gamma=gamma,
                            target_network_update_freq=q_network_iteration)
    agent = DQN(**model_parameters)
    agent.eval_net.load_state_dict(torch.load('./DQN_{}.pth'.format(env_name)))

    reward_sum = 0
    actions = []
    rewards = []

    for episode in range(10):
        state, _ = env.reset()
        for _ in range(2000):
            action = agent.act(state, infer=True)
            next_state, reward, done, truncated, info = env.step(action)

            actions.append(action)
            rewards.append(round(reward, 3))
            reward_sum += reward
            env.render()
            if done:
                break
            state = next_state
        print("*** Test reward", reward_sum)
        print("*** actions", actions)
        print("*** rewards", rewards, '\n')


if __name__ == '__main__':
    # run()
    eval()
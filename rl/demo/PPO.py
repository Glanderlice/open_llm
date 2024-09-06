from dataclasses import field, dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.rl_toolkit import get_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class PPOMemory:
    """
    PPO Memory 的特点：
    1.未来折扣奖励：reward和is_terminals共同完成,从后往前计算带折扣的奖励
    2.重要性采样：logprobs是用于此目的的数据
    3.适配PPO的损失计算公式：states和actions用于让新网络对已发生的历史进行概率评估、奖励估值
    综合看来, PPO的数据缓存更适合用分开的多个list存储
    """
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    is_terminals: list = field(default_factory=list)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory: PPOMemory = None):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)  # action_probs是softmax输出的概率值
        # 按概率分布采样：不能使用greedy策略取最大,必须进行按概率采样,否则动作序列会失去随机性, 并且模型会失去探索能力
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory is not None:
            # 缓存：输入的状态向量state,采取的action(由策略网络Πθ(at|st)生成+采样得到),以及action对应的概率密度log值(log_prob用于后续参数更新)
            memory.states.append(state)
            memory.actions.append(action)
            # action_log_prob相当于Π(ai|si)的对数值, 并且exp(dist.log_prob([0, 1, 2, 3]))得到的每个动作的概率密度,与softmax输出结果是一致的
            action_log_prob = dist.log_prob(action)
            memory.log_probs.append(action_log_prob)  # 这里对概率取log值是为了方便后续计算

        return action.item()  # 单值张量转标量：tensor->scalar

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        # action_log_prob = log(P(action|state)), 表示先求当前状态state下发生该action的概率(由网络Π(a|s)得到),再取对数值
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()  # 计算熵entropy = -Σ P(x)·logP(x), 其中Σ P(x)=1

        state_value = self.value_layer(state)  # 对当前状态进行估值

        return action_log_prob, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, k_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        # 用于参数更新的AC网络
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        # 用于与环境交互生成样本数据的AC网络, 它初始值与policy相同(policy.state_dict())
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []  # 计算每一步的带折扣奖励: rewards=[R1=r1+γ·R2, R2=r2+γ·R3,...R[T-1]=r[T-1]+γ·R[T],R[T]=r[T]]
        discounted_reward = 0
        # 从后往前算更方便, 每当遇到is_terminal,则按新的序列开始计算
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)  # 最终得到每个时间步t的带折扣奖励: R[t]=r[t]+γ·R[t+1]

        # 奖励做归一化处理 Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # 然后对奖励值做z-score归一化

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.log_probs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # 使用新网络Π 来评估old actions & values:
            # logprobs: logΠ(a|s), 表示当前state->action的概率的对数值
            # state_values: critic网络对当前state奖励估值
            # dist_entropy: 每个时间步的所有动作概率分布的熵
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 重要性采样：ratio = (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()  # 在s下执行a获得的实际奖励与state=s的奖励期望(可理解为奖励均值)之差, 我们希望它大于0
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # 损失第一项为PPO的clip版本的核心优化目标(Jθ), mse为了让critic估值更准确, dist_entropy为了降低actor预测的不确定性
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 用缓存的样本完成本次更新后, policy_old与新模型进行权重同步, 然后继续去与env交互生产数据
        self.policy_old.load_state_dict(self.policy.state_dict())  # Copy new weights into old policy


def run():
    render = False  # True将开启游戏画面
    target_reward = 230  # 轮均奖励值(停止训练条件) if accum_reward > target_reward * log_interval
    log_interval = 20  # 每玩多少轮打印一次log, 包括avg_reward per round, avg_len per round
    max_episodes = 20000  # max episodes, 最多玩这么多轮游戏
    max_timesteps = 300  # max timesteps per episode, 每轮游戏最多操作步数, 一旦达到就停止
    hidden_dim = 64  # 定义隐层神经网络节点数
    update_interval = 2000  # update policy, 每操作多少步进行一次模型更新
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor 未来奖励折扣因子
    k_epochs = 4  # update policy for K epochs 采集的数据会被学习这么多次
    eps_clip = 0.2  # clip parameter for PPO
    seed: Optional[int] = None

    # 观测到的状态空间(准确的说应该是observation)维度=8, 动作空间就4个动作可选
    env_name = "LunarLander-v2"
    env, state_dim, action_dim = get_env(env_name)

    if seed:
        torch.manual_seed(seed)

    memory = PPOMemory()
    agent = PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma, k_epochs, eps_clip)
    # print(lr,betas)

    # logging variables
    accum_reward = 0
    accum_length = 0
    step_counter = 0

    # training loop
    for episode in range(1, max_episodes + 1):  # 总共玩max_episodes轮游戏, episode表示一轮新游戏(达成游戏目标或操作超过步数上限本轮停止)
        state, _ = env.reset(seed=seed)  # 初始化(重新玩)
        episode_len, episode_reward = 0, 0.  # 本轮游戏走了多少步, 总共获得的奖励
        for _ in range(max_timesteps):  # 每轮游戏最多操作max_timesteps步,即单轮游戏序列长的上限(防止原地转圈或卡死等bug)
            step_counter += 1

            # 每一步都需要policy_old与env交互获得state状态向量,并作出action,然后将缓存到memory：
            action = agent.policy_old.act(state, memory)
            # 由游戏环境env反馈得到(新的状态,奖励,是否终止,是否出界,额外的调试信息)
            state, reward, done, truncated, _ = env.step(action)

            # 缓存奖励reward和终止信号is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_len += 1
            episode_reward += reward

            # 每操作update_timestep=2000步, 进行模型更新
            if step_counter % update_interval == 0:
                agent.update(memory)
                # 清空之前缓存的样本数据, 并重置time_step计数器
                memory.clear()  # 虽然重要性采样使得上一批可以再用,但前后两model不能差太远

            if render:
                env.render()
            if done:
                break  # 如果当前这轮游戏在max_timesteps步之前提前结束,则结束当前这轮
        # 累积本轮序列长,总奖励
        accum_length += episode_len
        accum_reward += episode_reward

        # 当最新的累积奖励高于预设阈值时,则停止训练
        if accum_reward > (log_interval * target_reward):
            print("训练目标已达成：平均奖励超过阈值")
            torch.save(agent.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # logging日志：会重置累积奖励和序列长度
        if episode % log_interval == 0:
            avg_length = int(accum_length / log_interval)  # 每一轮游戏平均玩多少步
            avg_reward = int((accum_reward / log_interval))  # (到目前为止)每一轮游戏的奖励均值

            print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
            accum_reward = 0
            accum_length = 0


def evaluate():
    max_episodes = 20000  # max episodes, 最多玩这么多轮游戏
    hidden_dim = 64  # 定义隐层神经网络节点数
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor 未来奖励折扣因子
    k_epochs = 4  # update policy for K epochs 采集的数据会被学习这么多次
    eps_clip = 0.2  # clip parameter for PPO

    # 观测到的状态空间(准确的说应该是observation)维度=8, 动作空间就4个动作可选
    env_name = "LunarLander-v2"
    env, state_dim, action_dim = get_env(env_name, render_mode="human")

    agent = PPO(state_dim, action_dim, hidden_dim, lr, betas, gamma, k_epochs, eps_clip)
    agent.policy_old.load_state_dict(torch.load('./PPO_{}.pth'.format(env_name)))


    # training loop
    for episode in range(1, max_episodes + 1):  # 总共玩max_episodes轮游戏, episode表示一轮新游戏(达成游戏目标或操作超过步数上限本轮停止)
        state, _ = env.reset()  # 初始化(重新玩)
        episode_len, episode_reward = 0, 0.  # 本轮游戏走了多少步, 总共获得的奖励
        for _ in range(2000):  # 每轮游戏最多操作max_timesteps步,即单轮游戏序列长的上限(防止原地转圈或卡死等bug)
            # 每一步都需要policy_old与env交互获得state状态向量,并作出action,然后将缓存到memory：
            action = agent.policy_old.act(state)
            # 由游戏环境env反馈得到(新的状态,奖励,是否终止,是否出界,额外的调试信息)
            state, reward, done, truncated, _ = env.step(action)

            episode_len += 1
            episode_reward += reward

            env.render()
            if done:
                break  # 如果当前这轮游戏在max_timesteps步之前提前结束,则结束当前这轮


if __name__ == '__main__':
    # run()  # PPO依然是一个on-policy算法, 因为它需要自己产生样本, 但相比传统梯度优化算法(模型产生的数据一旦模型发生参数更新,则不能再使用), 它训练效率更高(重要性采样)
    evaluate()

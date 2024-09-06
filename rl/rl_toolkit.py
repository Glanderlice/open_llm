import gym


def get_env(name, render_mode=None):
    env = gym.make(name, render_mode=render_mode)  # render_mode="human"
    # env = env.unwrapped
    state_dim = env.observation_space.shape[0]  # 2
    action_dim = env.action_space.n
    return env, state_dim, action_dim

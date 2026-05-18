import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def unwrap_env(env):
    real_env = env
    while hasattr(real_env, "env"):
        real_env = real_env.env
    return real_env

gym.register(
    id="FrozenLake-enhanced",
    entry_point="frozen_lake_enhanced:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,
)

def run(episodes, is_training=True, render=False):

    env = gym.make(
        'FrozenLake-enhanced',
        desc=None, map_name="8x8", is_slippery=True,
        render_mode='human' if render else None
    )

    real_env = unwrap_env(env)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate = 0.9
    gamma = 0.9
    epsilon = 1
    epsilon_decay = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):

        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:

            # Explore or exploit
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update
            if is_training:
                q[state, action] += learning_rate * (
                    reward + gamma * np.max(q[new_state]) - q[state, action]
                )

            state = new_state

        # -------------------------
        # END OF EPISODE RENDERING
        # -------------------------
        if render:
            if hasattr(real_env, "set_q"):
                real_env.set_q(q)
            if hasattr(real_env, "set_episode"):
                real_env.set_episode(i)
            real_env.render()

        # Decay ε
        epsilon = max(epsilon - epsilon_decay, 0)
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()


if __name__ == '__main__':
    run(1000, is_training=True, render=True)
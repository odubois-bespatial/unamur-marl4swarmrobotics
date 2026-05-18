import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

def state_to_xy(state, size=8):
    """Convertit un état 0–63 en coordonnées (x,y)."""
    return divmod(state, size)

def manhattan_distance(s, goal=(7, 7), size=8):
    x, y = state_to_xy(s, size)
    gx, gy = goal
    return abs(x - gx) + abs(y - gy)

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True,
                   render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1             
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    # Coefficient pour le shaping distance
    distance_factor = 0.1

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:

            # Choix action
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            old_distance = manhattan_distance(state)

            new_state, reward, terminated, truncated, _ = env.step(action)

            new_distance = manhattan_distance(new_state)

            # ----- REWARD SHAPING -----
            reward_distance = distance_factor * (old_distance - new_distance)
            reward += reward_distance
            # ---------------------------

            # Q-learning
            if is_training:
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        print(f"Episode: {i}   Last reward={reward:.2f}")

        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward > 0.9:  # goal atteint sans shaping
            rewards_per_episode[i] = 1

    env.close()

    # Graphique
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        with open("frozen_lake8x8.pkl", "wb") as f:
            pickle.dump(q, f)


if __name__ == '__main__':
    run(1000, is_training=True, render=True)
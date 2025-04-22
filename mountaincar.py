import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
import seaborn as sns

# Fixed parameters
EPISODES = 10000
FIXED_PARAMS = {
    'n_bins': 40,
    'alpha': 0.1,
    'gamma': 0.99,
    'epsilon': 0.1
}

def get_status(obs, env, n_bins):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_diff = (env_high - env_low) / n_bins
    pos = int((obs[0] - env_low[0]) / env_diff[0])
    vel = int((obs[1] - env_low[1]) / env_diff[1])
    return min(pos, n_bins - 1), min(vel, n_bins - 1)

def get_action(Q, obs, env, n_bins, epsilon):
    if np.random.rand() > epsilon:
        pos, vel = get_status(obs, env, n_bins)
        return np.argmax(Q[pos][vel])
    else:
        return np.random.choice([0, 1, 2])

def train_q_learning(param_name, values):
    results = []
    for val in tqdm(values, desc=f"Q-learning: {param_name}"):
        p = FIXED_PARAMS.copy()
        p[param_name] = val

        env = gym.make('MountainCar-v0')
        Q = np.zeros((p['n_bins'], p['n_bins'], env.action_space.n))
        rewards = []

        for episode in range(EPISODES):
            obs, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = get_action(Q, obs, env, p['n_bins'], p['epsilon'])
                next_obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward

                pos, vel = get_status(obs, env, p['n_bins'])
                next_pos, next_vel = get_status(next_obs, env, p['n_bins'])

                Q[pos][vel][action] += p['alpha'] * (
                    reward + p['gamma'] * np.max(Q[next_pos][next_vel]) - Q[pos][vel][action])

                obs = next_obs

            rewards.append(total_reward)
        results.append({"algorithm": "Q-learning", "param": param_name, "value": val, "rewards": rewards})
    return results

def train_sarsa():
    p = FIXED_PARAMS.copy()
    env = gym.make('MountainCar-v0')
    Q = np.zeros((p['n_bins'], p['n_bins'], env.action_space.n))
    rewards = []

    for episode in tqdm(range(EPISODES), desc="SARSA"):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        action = get_action(Q, obs, env, p['n_bins'], p['epsilon'])

        while not done:
            next_obs, reward, done, truncated, _ = env.step(action)
            next_action = get_action(Q, next_obs, env, p['n_bins'], p['epsilon'])

            pos, vel = get_status(obs, env, p['n_bins'])
            next_pos, next_vel = get_status(next_obs, env, p['n_bins'])

            Q[pos][vel][action] += p['alpha'] * (
                reward + p['gamma'] * Q[next_pos][next_vel][next_action] - Q[pos][vel][action])

            obs = next_obs
            action = next_action
            total_reward += reward

        rewards.append(total_reward)
    return [{"algorithm": "SARSA", "param": "algorithm", "value": "SARSA", "rewards": rewards}]

def train_q_learning_fixed():
    p = FIXED_PARAMS.copy()
    env = gym.make('MountainCar-v0')
    Q = np.zeros((p['n_bins'], p['n_bins'], env.action_space.n))
    rewards = []

    for episode in tqdm(range(EPISODES), desc="Q-learning (fixed)"):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = get_action(Q, obs, env, p['n_bins'], p['epsilon'])
            next_obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            pos, vel = get_status(obs, env, p['n_bins'])
            next_pos, next_vel = get_status(next_obs, env, p['n_bins'])

            Q[pos][vel][action] += p['alpha'] * (
                reward + p['gamma'] * np.max(Q[next_pos][next_vel]) - Q[pos][vel][action])

            obs = next_obs

        rewards.append(total_reward)
    
    env.close()
    
    # evaluate environment
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    env = RecordVideo(env, "videos", name_prefix="mountaincar")
    obs, _ = env.reset()
    done = False
    while not done:
        action = get_action(Q, obs, env, p['n_bins'], p['epsilon'])
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = next_obs
        done = terminated or truncated
        if done:
            if truncated:
                print("Episode ended due to truncation.")
            else:
                print("Episode ended due to termination.")
    env.close()

    return [{"algorithm": "Q-learning-fixed", "param": "algorithm", "value": "Q-learning-fixed", "rewards": rewards}]

if __name__ == "__main__":
    # Comparison Experiments
    all_results = []
    all_results += train_q_learning("n_bins", [10, 20, 40, 60])
    all_results += train_q_learning("alpha", [0.01, 0.1, 0.5])
    all_results += train_q_learning("gamma", [0.7, 0.99])
    all_results += train_q_learning("epsilon", [0.01, 0.1, 0.5])
    all_results += train_sarsa()
    all_results += train_q_learning_fixed()

    # create DataFrame for plotting
    data = []
    for res in all_results:
        for ep, r in enumerate(res["rewards"]):
            data.append({"algorithm": res["algorithm"], "param": res["param"], "value": res["value"], "episode": ep, "reward": r})

    df = pd.DataFrame(data)

    # Process the data for plotting
    os.makedirs("figures", exist_ok=True)
    sns.set(style="darkgrid")

    # Comparison graphs for each parameter
    for param in df["param"].unique():
        if param != "algorithm":
            subset = df[(df["param"] == param) & (df["episode"] % 100 == 0)]
            plt.figure(figsize=(10, 5))
            plt.ylim(-1000, 0)
            sns.lineplot(data=subset, x="episode", y="reward", hue="value")
            plt.title(f"Comparison by {param}")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend(title=param)
            plt.tight_layout()
            plt.savefig(f"figures/comparison_{param}.png")
            plt.close()

    # SARSA vs Q-learning-fixed graph
    algo_subset = df[df["param"] == "algorithm"]
    plt.figure(figsize=(10, 5))
    plt.ylim(-1000, 0)
    sns.lineplot(data=algo_subset[algo_subset["episode"] % 100 == 0], x="episode", y="reward", hue="value")
    plt.title("SARSA vs Q-learning (fixed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig("figures/comparison_sarsa_vs_qlearning_fixed.png")
    plt.close()
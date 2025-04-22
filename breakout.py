import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import matplotlib.pyplot as plt
from collections import deque

# --- Hyperparameters ---
BATCH_SIZE = 32
TOTAL_STEPS = 10_000_000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 500_000
BUFFER_SIZE = 100_000
GAMMA = 0.99
TRAINING_START_IT = 80_000
UPDATE_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 10_000
LEARNING_RATE = 1.25e-4
EVAL_EPISODES = 30
SAVE_INTERVAL = 100_000
MODEL_DIR = "models"
VIDEO_DIR = "videos"
# ----------------------
# --- Create directories for saving models and videos ---
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256), nn.ReLU(),
            nn.Linear(256, nb_actions)
        )

    def forward(self, x):
        return self.network(x / 255.)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, next_state, action, reward, done):
        self.buffer.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, next_states, actions, rewards, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# --- Main training loop ---
def deep_Q_learning(env, device):
    rb = ReplayBuffer(capacity=BUFFER_SIZE)
    q_network = DQN(env.action_space.n).to(device)
    target_q_network = DQN(env.action_space.n).to(device)
    target_q_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    smoothed_rewards = []
    rewards = []
    max_reward = float('-inf')
    step = 0
    progress_bar = tqdm(total=TOTAL_STEPS)

    while step <= TOTAL_STEPS:
        state, _ = env.reset()
        total_rewards = 0
        done = False

        while not done and step <= TOTAL_STEPS:
            epsilon = max(EPSILON_END, EPSILON_START - step / EPSILON_DECAY_STEPS * (EPSILON_START - EPSILON_END))

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
                    q = q_network(state_tensor)
                    action = torch.argmax(q, dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward
            reward = np.sign(reward)

            rb.add(state, next_state, action, reward, done)

            # Store the reward for plotting
            if step > TRAINING_START_IT and step % UPDATE_FREQUENCY == 0:
                states, next_states, actions, rewards_tensor, dones_tensor = rb.sample(BATCH_SIZE)
                states, next_states = states.to(device), next_states.to(device)
                actions = actions.to(device)
                rewards_tensor = rewards_tensor.to(device)
                dones_tensor = dones_tensor.to(device)

                with torch.no_grad():
                    max_next_q = target_q_network(next_states).max(dim=1)[0]
                    target = rewards_tensor + GAMMA * max_next_q * (1 - dones_tensor)

                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = torch.nn.functional.huber_loss(current_q, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if step % SAVE_INTERVAL == 0 and step >= 0:
                model_path = os.path.join(MODEL_DIR, f"q_network_step{step}.pt")
                torch.save(q_network.state_dict(), model_path)

            if step % TARGET_UPDATE_FREQUENCY == 0:
                target_q_network.load_state_dict(q_network.state_dict())

            state = next_state
            step += 1
            progress_bar.update(1)

        rewards.append(total_rewards)

# --- Evaluate the model ---
def evaluate(env, model_path, device, epsilon=0.05):
    q_network = DQN(env.action_space.n).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

    total_rewards = []
    for ep in range(EVAL_EPISODES):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
                    action = q_network(state_tensor).argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        # print(f"Episode {ep+1}: Reward = {episode_reward}")
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {EVAL_EPISODES} episodes: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    # train
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = FrameStackObservation(env, stack_size=4)
    env = RecordVideo(env, VIDEO_DIR, name_prefix="breakout", episode_trigger=lambda x: x % 250 == 0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deep_Q_learning(env, device)

    # evaluate
    rewards = []
    steps = list(range(0, 10000001, 100000))
    for i in steps:
        print(f"Evaluating model at step {i/1e4}...")
        eval_env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)
        eval_env = AtariPreprocessing(eval_env, grayscale_obs=True, frame_skip=4, scale_obs=False)
        eval_env = FrameStackObservation(eval_env, stack_size=4)
        avg_reward = evaluate(eval_env, model_path=os.path.join(MODEL_DIR, f"q_network_step{i}.pt"), device=device)
        rewards.append(avg_reward)
        eval_env.close()
    steps_million = [x / 1e6 for x in steps]
    plt.plot(steps_million, rewards)
    plt.title('Average Reward on Breakout')
    plt.xlabel('Training Steps (Million)')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('average_rewards_target_dqn.png')
    plt.close()
    
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from tqdm import tqdm
import os

# --- Konfiguration & Hyperparametre ---
ENV_NAME = "LunarLander-v2"
MODEL_PATH = "models/dqn_lunarlander.pth"
LEARNING_RATE = 0.001
GAMMA = 0.99            # Diskonteringsfaktor
BATCH_SIZE = 64
BUFFER_SIZE = 100000    # Replay Buffer størrelse
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
SYNC_TARGET_EPISODES = 100 # Opdater target netværk hver N episode
NUM_EPISODES = 1000

# Vælg enhed (CPU, CUDA eller MPS til Mac Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Bruger enhed: {device}")

# --- 1. Neural Network Klasse (Q-Network) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Experience Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Gem oplevelse som tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)

# --- 3. DQN Agent Klasse ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        
        # Opret Q-Netværk og Target Netværk
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        
        # Kopier vægte til target netværk (start synkroniseret)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target net skal ikke trænes direkte

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        # Epsilon-Greedy strategi
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return # Vent til vi har nok data

        # Hent batch fra replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Konverter til tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Beregn nuværende Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Beregn Target Q(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        # Beregn loss og opdater vægte
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(" -> Target netværk synkroniseret.")

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model gemt til {path}")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.policy_net.eval()
        print(f"Model indlæst fra {path}")

# --- 4. Main Løkke (Træning og Test) ---
def train():
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    print(f"Starter træning på {ENV_NAME} i {NUM_EPISODES} episoder...")
    
    scores = []
    
    for episode in tqdm(range(1, NUM_EPISODES + 1)):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            score += reward
        
        # Opdater Epsilon (Decay)
        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        scores.append(score)

        # Synkroniser target netværk
        if episode % SYNC_TARGET_EPISODES == 0:
            agent.update_target_network()

        # Log fremskridt hver 100. episode
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            tqdm.write(f"Episode {episode} | Avg Score (100): {avg_score:.2f} | Epsilon: {agent.epsilon:.4f}")

    agent.save(MODEL_PATH)
    env.close()

def test():
    env = gym.make(ENV_NAME, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        agent.epsilon = 0.0 # Ingen tilfældighed i test
    else:
        print("Ingen gemt model fundet. Kør træning først!")
        return

    print("Starter test visning (Tryk Ctrl+C for at stoppe)...")
    for i in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Test Episode {i+1}: Total Reward: {total_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    # Tjek om vi skal træne eller teste
    if os.path.exists(MODEL_PATH):
        print(f"Fandt eksisterende model: {MODEL_PATH}")
        choice = input("Vil du (t)este den eksisterende model eller (T)ræne en ny? (t/T): ").lower()
        if choice == 't':
            test()
        else:
            train()
            test()
    else:
        train()
        test()

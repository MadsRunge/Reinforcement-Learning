"""
Q-Learning implementation for Taxi-v3 environment from Gymnasium
"""

import gymnasium as gym
import numpy as np
import random

# Hyperparametre
LEARNING_RATE = 0.1  # alpha
DISCOUNT_FACTOR = 0.6  # gamma
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # Epsilon aftager eksponentielt
NUM_EPISODES = 25000

# Initialiser miljøet
env = gym.make("Taxi-v3")

# Initialiser Q-tabel
# Taxi-v3 har 500 mulige tilstande og 6 mulige handlinger
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

print(f"Tilstandsrum størrelse: {state_space_size}")
print(f"Handlingsrum størrelse: {action_space_size}")
print(f"Q-tabel shape: {q_table.shape}")
print("\nStarter træning...\n")


def epsilon_greedy_policy(state, epsilon):
    """
    Vælg en handling baseret på Epsilon-Greedy strategi

    Args:
        state: Den aktuelle tilstand
        epsilon: Sandsynligheden for at vælge en tilfældig handling

    Returns:
        action: Den valgte handling
    """
    if random.uniform(0, 1) < epsilon:
        # Exploration: Vælg en tilfældig handling
        return env.action_space.sample()
    else:
        # Exploitation: Vælg den bedste handling baseret på Q-tabel
        return np.argmax(q_table[state, :])


# Træningsløkke
epsilon = EPSILON_START
rewards_per_episode = []

for episode in range(NUM_EPISODES):
    # Reset miljøet og få den første tilstand
    state, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Vælg en handling ved hjælp af Epsilon-Greedy strategi
        action = epsilon_greedy_policy(state, epsilon)

        # Udfør handlingen i miljøet
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-Learning opdateringsformel:
        # Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])

        new_value = old_value + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max - old_value
        )

        # Opdater Q-tabel
        q_table[state, action] = new_value

        # Opdater tilstand og total reward
        state = next_state
        total_reward += reward

    # Gem total reward for denne episode
    rewards_per_episode.append(total_reward)

    # Aftag epsilon (Exploration rate)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Print fremskridt hver 1000. episode
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-1000:])
        print(f"Episode {episode + 1}/{NUM_EPISODES} - "
              f"Gennemsnitlig reward (sidste 1000): {avg_reward:.2f} - "
              f"Epsilon: {epsilon:.4f}")

print("\nTræning afsluttet!")

# Beregn og vis statistikker
print("\n--- Træningsstatistikker ---")
print(f"Gennemsnitlig reward (alle episoder): {np.mean(rewards_per_episode):.2f}")
print(f"Gennemsnitlig reward (sidste 1000 episoder): {np.mean(rewards_per_episode[-1000:]):.2f}")
print(f"Gennemsnitlig reward (første 1000 episoder): {np.mean(rewards_per_episode[:1000]):.2f}")
print(f"Maksimal reward: {np.max(rewards_per_episode):.2f}")
print(f"Minimal reward: {np.min(rewards_per_episode):.2f}")


# Test den trænede agent
print("\n--- Test af trænet agent ---")
num_test_episodes = 10
test_rewards = []

env_render = gym.make("Taxi-v3", render_mode="ansi")

for test_episode in range(num_test_episodes):
    state, info = env_render.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        # Brug kun exploitation (epsilon = 0)
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, info = env_render.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state
        steps += 1

    test_rewards.append(total_reward)
    print(f"Test episode {test_episode + 1}: Reward = {total_reward}, Steps = {steps}")

print(f"\nGennemsnitlig test reward: {np.mean(test_rewards):.2f}")

# Luk miljøet
env.close()
env_render.close()

print("\nFærdig!")

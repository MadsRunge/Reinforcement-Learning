"""
Q-Learning implementation for Taxi-v3 environment from Gymnasium
Inkluderer gem/indlæs funktionalitet og pygame visualisering
"""

import gymnasium as gym
import numpy as np
import random
import os
import time

# Hyperparametre
LEARNING_RATE = 0.1  # alpha
DISCOUNT_FACTOR = 0.6  # gamma
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995  # Epsilon aftager eksponentielt
NUM_EPISODES = 25000

# Filnavn til gem/indlæs af Q-tabel
Q_TABLE_FILE = "models/q_table_taxi.npy"

# Initialiser miljøet
env = gym.make("Taxi-v3")

# Initialiser Q-tabel
# Taxi-v3 har 500 mulige tilstande og 6 mulige handlinger
state_space_size = env.observation_space.n
action_space_size = env.action_space.n

print(f"Tilstandsrum størrelse: {state_space_size}")
print(f"Handlingsrum størrelse: {action_space_size}")
print()

# Tjek om der findes en gemt Q-tabel
if os.path.exists(Q_TABLE_FILE):
    print(f"✓ Fandt gemt Q-tabel: {Q_TABLE_FILE}")
    q_table = np.load(Q_TABLE_FILE)
    print(f"✓ Q-tabel indlæst med shape: {q_table.shape}")
    print("→ Springer træning over og går direkte til test\n")
    skip_training = True
else:
    print(f"✗ Ingen gemt Q-tabel fundet ({Q_TABLE_FILE})")
    print("→ Starter fuld træning fra bunden\n")
    q_table = np.zeros((state_space_size, action_space_size))
    skip_training = False


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
if not skip_training:
    print("=== STARTER TRÆNING ===\n")
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

    print("\n✓ Træning afsluttet!")

    # Gem Q-tabel til fil
    np.save(Q_TABLE_FILE, q_table)
    print(f"✓ Q-tabel gemt til: {Q_TABLE_FILE}")

    # Beregn og vis statistikker
    print("\n--- Træningsstatistikker ---")
    print(f"Gennemsnitlig reward (alle episoder): {np.mean(rewards_per_episode):.2f}")
    print(f"Gennemsnitlig reward (sidste 1000 episoder): {np.mean(rewards_per_episode[-1000:]):.2f}")
    print(f"Gennemsnitlig reward (første 1000 episoder): {np.mean(rewards_per_episode[:1000]):.2f}")
    print(f"Maksimal reward: {np.max(rewards_per_episode):.2f}")
    print(f"Minimal reward: {np.min(rewards_per_episode):.2f}")
else:
    print("=== TRÆNING SPRUNGET OVER (BRUGER GEMT Q-TABEL) ===\n")


# Test den trænede agent med pygame visualisering
print("\n--- Test af trænet agent med visualisering ---")
print("(Pygame vindue åbnes - luk det for at stoppe)\n")

num_test_episodes = 10
test_rewards = []

# Opret miljø med human rendering (pygame)
env_render = gym.make("Taxi-v3", render_mode="human")

for test_episode in range(num_test_episodes):
    state, info = env_render.reset()
    total_reward = 0
    done = False
    steps = 0

    print(f"\n→ Test episode {test_episode + 1}/{num_test_episodes} starter...")

    while not done:
        # Brug kun exploitation (epsilon = 0)
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, info = env_render.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state
        steps += 1

        # Tilføj forsinkelse for at gøre bevægelser synlige
        time.sleep(0.1)

    test_rewards.append(total_reward)
    print(f"  ✓ Episode {test_episode + 1}: Reward = {total_reward}, Steps = {steps}")

    # Kort pause mellem episodes
    time.sleep(0.5)

print(f"\n--- Test Resultat ---")
print(f"Gennemsnitlig test reward: {np.mean(test_rewards):.2f}")
print(f"Bedste test reward: {np.max(test_rewards):.2f}")
print(f"Værste test reward: {np.min(test_rewards):.2f}")

# Luk miljøet
env.close()
env_render.close()

print("\n✓ Færdig!")
print(f"\nNæste gang du kører scriptet, vil den gemte Q-tabel ({Q_TABLE_FILE}) blive indlæst automatisk.")
print("For at træne igen, slet filen først: rm q_table_taxi.npy")

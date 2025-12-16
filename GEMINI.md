# Reinforcement Learning - Taxi-v3 Q-Learning

This project implements the Q-Learning algorithm to solve the "Taxi-v3" environment from the Gymnasium library. It features a complete pipeline for training an agent, persisting the learned knowledge, and visualizing the agent's performance.

## Project Overview

The core of the project is a Q-Learning agent that learns to navigate a grid world to pick up and drop off passengers.

*   **Algorithm**: Q-Learning (Off-policy TD control).
*   **Environment**: `Taxi-v3` (Gymnasium).
*   **Key Features**:
    *   **Training**: Runs for 25,000 episodes with decaying epsilon-greedy exploration.
    *   **Persistence**: Saves the learned Q-table to `q_table_taxi.npy`. If this file exists, training is skipped.
    *   **Visualization**: Uses Pygame to render the agent's actions after training (or loading a saved model).
    *   **Language**: Source code comments and CLI output are in **Danish**.

## Project Structure

*   `q_learning_taxi.py`: The main entry point. Contains the Q-Learning algorithm, training loop, and testing/visualization logic.
*   `setup.sh`: A bash script to automate the environment setup using `pyenv` and `pyenv-virtualenv`.
*   `requirements.txt`: Python dependencies (`gymnasium`, `numpy`, `pygame`).
*   `README.md`: Project documentation in Danish.

## Setup and Installation

The project includes a `setup.sh` script to streamline the installation process, specifically designed for macOS/Linux users with `pyenv`.

**Automated Setup:**
```bash
chmod +x setup.sh
./setup.sh
```
This script will:
1.  Check for `pyenv`.
2.  Install Python 3.11.9 if missing.
3.  Create a virtual environment named `rl_taxi_env`.
4.  Activate the environment locally (creating `.python-version`).
5.  Install dependencies from `requirements.txt`.

**Manual Setup:**
If you prefer managing your own environment:
```bash
pip install -r requirements.txt
```

**Troubleshooting:**
*   **Error `pyenv: no such command 'virtualenv'`**:
    *   Missing `pyenv-virtualenv` plugin.
    *   Fix: `brew install pyenv-virtualenv` AND run `eval "$(pyenv virtualenv-init -)"`.

## Usage

**Run the Agent:**
```bash
python q_learning_taxi.py
```

**Behavior:**
1.  **Check for existing model**: The script checks for `q_table_taxi.npy`.
    *   **If found**: Skips training, loads the Q-table, and proceeds directly to the visualization test.
    *   **If not found**: Starts training from scratch (25,000 episodes), saves the Q-table upon completion, and then runs the test.
2.  **Retraining**: To force the agent to retrain, delete the saved Q-table:
    ```bash
    rm q_table_taxi.npy
    ```

## Development Conventions

*   **Language**: Code comments and string outputs are written in **Danish**.
*   **Style**: Standard Python conventions (PEP 8).
*   **Dependencies**: Strict version pinning is used in `requirements.txt` to ensure reproducibility (Gymnasium 0.29.1, NumPy 1.26.4).

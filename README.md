# Air Hockey RL

A reinforcement learning agent that learns to play air hockey through self-play training.

## Demo

https://github.com/user-attachments/assets/88428ee6-d9bb-448b-b3c7-fcb904a3bf0f


## Features

- **Play against the AI** - Challenge the trained agent using keyboard controls
- **Self-play training** - Agent improves by playing against past versions of itself
- **TensorBoard logging** - Monitor training progress in real-time

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/air_hockey_rl.git
cd air_hockey_rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Play Against the AI

```bash
python play.py
```

Controls:
- **W** - Move up
- **S** - Move down
- **A** - Move left
- **D** - Move right
- **Diagonals** - Combine keys (WA, WD, SA, SD)
- **Q/ESC** - Quit

### Train Your Own Agent

```bash
python train.py
```

Training parameters can be adjusted in `train.py`. Default training runs for 2M timesteps with 8 parallel environments.

### Watch AI vs AI

```bash
python test.py
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=./logs
```

Then open http://localhost:6006 in your browser.

## Project Structure

```
air_hockey_rl/
├── game.py          # Core game physics and rendering
├── env.py           # Gymnasium environment wrapper
├── opponent.py      # Opponent controllers (rule-based, model-based, self-play)
├── train.py         # Self-play training script
├── play.py          # Human vs AI gameplay
├── test.py          # AI vs AI evaluation
└── requirements.txt # Python dependencies
```

## How It Works

### Game Mechanics
- 2D air hockey with physics-based puck and paddle movement
- 9 discrete actions: stay, 4 cardinal directions, 4 diagonals
- First to 5 goals wins

### Training
The agent uses **Proximal Policy Optimization (PPO)** with self-play:
1. Agent starts by playing against a simple rule-based opponent
2. Periodically saves checkpoints of its policy
3. Plays against older versions of itself to continuously improve
4. Learns both offensive and defensive strategies

### Reward Structure
- **+1.0** for scoring a goal
- **-1.0** for conceding a goal
- **+0.1** for hitting puck toward opponent's goal
- **-0.05** for hitting puck toward own goal
- **+0.01** for staying close to puck when it's in agent's half (defensive positioning)

## Requirements

- Python 3.8+
- pygame
- gymnasium
- stable-baselines3
- numpy
- tensorboard

## License

MIT License

# RL-Powered Bio-Farm Optimization (MLOps)

## Overview
This project simulates an indoor vertical bio-farm and utilizes Deep Reinforcement Learning to optimize resource consumption. It trains an autonomous AI agent to find the mathematical balance between maximizing plant growth and minimizing electricity and water usage, aligning with SDG 12 and SDG 13.

## 🗂️ Project Structure
\`\`\`text
biofarm_ai/
│
├── data/                     # Configuration and dynamic simulation parameters
├── models/                   # Saved RL models (ppo_model.zip)
├── src/                      # Source Code
│   ├── environment.py        # Custom Gymnasium Environment MDP
│   └── train.py              # RL agent training and evaluation script
├── visualization/            # Dashboard and UI
│   └── dashboard.py          # Streamlit interactive application
├── requirements.txt          # Project dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project Documentation
\`\`\`

## Features
- Simulates a Continuous Markov Decision Process (MDP) for agricultural resource management.
- Trains an autonomous decision-making model using state-of-the-art Reinforcement Learning.
- Visualizes real-time sensor data and agent actions via an interactive dashboard.
- Implements a clean MLOps architectural structure (`src/` and `visualization/` separation).

## Requirements
- Python 3.10+
- Stable-Baselines3
- Gymnasium
- NumPy
- Streamlit

## Model Evaluation
The Reinforcement Learning agent is evaluated using **Mean Final Reward** and **Standard Deviation** over multiple evaluation episodes. The reward function heavily penalizes excess resource expenditure (water, nutrients, LED power) while rewarding incremental plant growth.

## Model
The core intelligence is powered by a **Proximal Policy Optimization (PPO)** algorithm. It utilizes an Actor-Critic Multi-Layer Perceptron (MLP) neural network architecture to process the continuous sensor data and output deterministic hardware commands.

## Visualization
The farm's data and the agent's actions are visualized using a **Streamlit** dashboard. It provides live metric tracking for the 5 key sensors and an automated logging window to monitor the exact hardware commands the AI is issuing every simulated day.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
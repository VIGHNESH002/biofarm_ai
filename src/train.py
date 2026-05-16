import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment import BioFarmEnv

# --- VERSION CONTROL ---
MODEL_VERSION = "v1.0"  
TIMESTEPS = 50000

env = BioFarmEnv()
print(f"🧠 Initializing PPO Model ({MODEL_VERSION})...")
model = PPO("MlpPolicy", env, verbose=1)

print(f"🚀 Training Model for {TIMESTEPS} timesteps...")
model.learn(total_timesteps=TIMESTEPS)

# Save with the version tag
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", f"ppo_model_{MODEL_VERSION}")
model.save(model_path)
print(f"✅ Model saved successfully to {model_path}.zip")

print("\n📊 Evaluating the Trained AI...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Final Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
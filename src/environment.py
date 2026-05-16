import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BioFarmEnv(gym.Env):
    def __init__(self):
        super(BioFarmEnv, self).__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.max_steps = 60 
        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([0.0, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        water, nutrients, light = action

        current_moisture = self.state[1]
        current_nutrients = self.state[2]

        # --- SMART PHYSICS ---
        # The plant grows based on the SOIL conditions, not just the pump being on.
        growth_rate = 0.05 * current_moisture * current_nutrients * light
        
        # If the AI over-waters or over-fertilizes (> 0.85), the plant drowns and stops growing!
        if current_moisture > 0.85 or current_nutrients > 0.85:
            growth_rate = 0.0 

        new_growth = np.clip(self.state[0] + growth_rate, 0, 1)
        new_moisture = np.clip(current_moisture + water - 0.15, 0, 1)  # Natural evaporation
        new_nutrients = np.clip(current_nutrients + nutrients - 0.15, 0, 1) # Depletion
        new_temp = np.clip(self.state[3] + (light * 0.2) - 0.1, 0, 1) 
        
        self.state = np.array([new_growth, new_moisture, new_nutrients, new_temp, light], dtype=np.float32)

        # --- Reward Logic ---
        resource_cost = (water * 0.1) + (nutrients * 0.2) + (light * 0.5)
        reward = (growth_rate * 100) - resource_cost

        terminated = bool(self.state[0] >= 1.0) 
        truncated = bool(self.current_step >= self.max_steps) 

        info = {'resource_cost': float(resource_cost)}
        
        return self.state, float(reward), terminated, truncated, info
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from .custom_policy import CustomActorCriticPolicy

class PPOAgent:
    def __init__(self, env, model_path="ppo_apex_model"):
        self.env = DummyVecEnv([lambda: env])
        self.model_path = model_path
        self.model = None
    def create_model(self, verbose=1):
        """
        Initialize a new PPO model with Custom Policy.
        """
        self.model = PPO(
            CustomActorCriticPolicy, 
            self.env, 
            verbose=verbose,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1, # Force EXTREME exploration
        )

    def train(self, total_timesteps=10000):
        """
        Train the model.
        """
        if self.model is None:
            self.create_model()
        self.model.learn(total_timesteps=total_timesteps)
        self.save()

    def predict(self, observation):
        """
        Predict action for a given observation.
        """
        if self.model is None:
            self.load() # Try to load if not in memory
        # Get action, value, and log_prob
        # We need to use the policy directly to get the value and log_prob
        obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
        distribution = self.model.policy.get_distribution(obs_tensor)
        actions = distribution.get_actions(deterministic=True)
        log_prob = distribution.log_prob(actions)
        values = self.model.policy.predict_values(obs_tensor)
        
        return {
            "action": actions.detach().cpu().numpy()[0],
            "value": values.detach().cpu().numpy()[0][0],
            "log_prob": log_prob.detach().cpu().numpy()[0]
        }

    def save(self):
        self.model.save(self.model_path)

    def load(self):
        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            print("Model not found, creating new one.")
            self.create_model()

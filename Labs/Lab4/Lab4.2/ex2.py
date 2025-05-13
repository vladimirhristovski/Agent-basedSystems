import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("LunarLanderContinuous-v3", render_mode="rgb_array"))
env.reset()

n_actions = env.action_space.shape[0]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG(
    "MlpPolicy",
    env=env,
    learning_rate=0.001,
    buffer_size=100000,
    batch_size=64,
    train_freq=(5, 'episode'),
    device='cuda',
    action_noise=action_noise,
    verbose=1,
)

model.learn(total_timesteps=100000)

mean_reward_50, _ = evaluate_policy(model, env, n_eval_episodes=50)
print(f"Average reward for 50 episodes: {mean_reward_50:.2f}")

mean_reward_100, _ = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Average reward for 100 episodes: {mean_reward_100:.2f}")

render_env = Monitor(gym.make("LunarLanderContinuous-v3", render_mode="human"))
obs, _ = render_env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = render_env.step(action)
    time.sleep(0.02)

render_env.close()

# prosecnata nagrada za 50 iteracii e -151.7 odnosno -181.99 za 100 iteracii

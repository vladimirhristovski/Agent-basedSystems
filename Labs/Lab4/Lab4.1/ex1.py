import gymnasium as gym
import numpy as np
from deep_q_learning import DDPG, OrnsteinUhlenbeckActionNoise


def evaluate_agent(agent, env, episodes=50):
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state, discrete=False)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, terminated, info, _ = env.step(action)
            done = terminated
            total_reward += reward

        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Evaluation over {episodes} episodes: Avg Reward = {avg_reward:.2f}")
    return avg_reward


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env.reset()

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    agent = DDPG(
        state_space_shape=state_dim,
        action_space_shape=action_dim,
        learning_rate_actor=0.001,
        learning_rate_critic=0.002,
        discount_factor=0.9,
        batch_size=64,
        memory_size=100000
    )

    noise = OrnsteinUhlenbeckActionNoise(action_space_shape=action_dim)

    num_episodes = 3000
    max_steps_per_episode = 3000

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        noise.reset()

        for step in range(max_steps_per_episode):
            action = agent.get_action(state, discrete=False) + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, info, _ = env.step(action)
            done = terminated

            agent.update_memory(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.train()

        if (episode + 1) % 20 == 0:
            agent.update_target_model()

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    # Evaluation after training
    print("\nEvaluating trained agent...")
    evaluate_agent(agent, env, episodes=50)
    evaluate_agent(agent, env, episodes=100)

# so ovie postaveni parametri se dojde do prosecna nagrada za 50 iteracii -802,46 odnosno -810,49 za 100 iteracii

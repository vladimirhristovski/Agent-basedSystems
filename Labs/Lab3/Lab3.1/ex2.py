import gymnasium as gym
import torch.nn as nn
import numpy as np
from deep_q_learning import DDQN


def build_model(state_space_shape, num_actions):
    return nn.Sequential(
        nn.Linear(state_space_shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, num_actions)
    )


def train_dqn_agent(env, agent, num_episodes=3000, max_steps=3000, epsilon_decay=0.999):
    epsilon = 1.0
    epsilon_min = 0.01
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, epsilon)
            next_state, reward, done, info, _ = env.step(action)

            agent.update_memory(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.train()

        if episode % 5 == 0:
            agent.update_target_model()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} | Reward: {total_reward} | Epsilon: {epsilon:.3f}")

    return rewards


def test_agent(env, agent, num_episodes=3000, max_steps=3000):
    total_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, epsilon=0.0)
            next_state, reward, done, info, _ = env.step(action)
            env.render()

            state = next_state
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)
        print(f"[Test] Episode {ep + 1}: Reward = {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {num_episodes} test episodes: {avg_reward:.2f}\n")
    return avg_reward


if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    state_space_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = build_model(state_space_shape, num_actions)
    target_model = build_model(state_space_shape, num_actions)

    agent = DDQN(
        state_space_shape=state_space_shape,
        num_actions=num_actions,
        model=model,
        target_model=target_model,
        learning_rate=0.01,
        discount_factor=0.995,
        batch_size=64,
        memory_size=2048
    )

    print("\nTraining agent\n")
    training_rewards = train_dqn_agent(env, agent)

    print("\nTesting agent with 50 episodes")
    avg_reward_50 = test_agent(env, agent, num_episodes=50)

    print("\nTesting agent with 100 episodes")
    avg_reward_100 = test_agent(env, agent, num_episodes=100)

    env.close()

    print("Final Summary:")
    print(f"Average reward (50 episodes): {avg_reward_50:.2f}")
    print(f"Average reward (100 episodes): {avg_reward_100:.2f}")

# prosecnata nagrada e -3000, odnosno nikogas ne stigna do celta. Se dobivaat isti rezultati kako vo lab2, odnosno ne se stignuva do celta

import gym
import torch.nn as nn
import numpy as np
from deep_q_learning import DQN
from PIL import Image


def preprocess_state(state):
    img = Image.fromarray(state)
    img = img.convert('L')
    img = img.resize((84, 84))
    grayscale_img = np.array(img, dtype=np.float32)
    grayscale_img = grayscale_img / 255.0
    return grayscale_img[np.newaxis, :, :]


def build_model(state_space_shape, num_actions):
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=8, stride=4),  # Conv layer 1
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Conv layer 2
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Conv layer 3
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, num_actions)
    )


def train_dqn_agent(env, agent, num_episodes=3000, max_steps=3000, epsilon_decay=0.999):
    epsilon = 1.0
    epsilon_min = 0.01
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        preprocessed_state = preprocess_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(preprocessed_state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            preprocessed_next_state = preprocess_state(next_state)
            agent.update_memory(preprocessed_state, action, reward, preprocessed_next_state, done)

            preprocessed_state = preprocessed_next_state
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
        preprocessed_state = preprocess_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(preprocessed_state, epsilon=0.0)
            next_state, reward, done, _, _ = env.step(action)
            preprocessed_state = preprocess_state(next_state)
            env.render()

            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)
        print(f"[Test] Episode {ep + 1}: Reward = {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage reward over {num_episodes} test episodes: {avg_reward:.2f}\n")
    return avg_reward


if __name__ == '__main__':
    env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
    state_space_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = build_model(state_space_shape, num_actions)
    target_model = build_model(state_space_shape, num_actions)

    agent = DQN(
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

# prosecnata nagrada pri testiranje vo site epizodi beshe 210, toa e spored najdobrata akcija koja sto ja naucil agnetot
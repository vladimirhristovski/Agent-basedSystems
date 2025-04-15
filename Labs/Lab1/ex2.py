import numpy as np
import gymnasium as gym
from mdp import policy_iteration

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi').unwrapped

    state, _ = env.reset()
    env.render()
    terminated = False
    max_reward = ...
    best_discount = 0.5

    print("Total reward per discount:")
    for discount in [0.5, 0.7, 0.9]:
        state, _ = env.reset()
        terminated = False
        policy, _ = policy_iteration(env, env.action_space.n, env.observation_space.n, discount_factor=discount)
        total_reward = ...
        while not terminated:
            new_action = np.argmax(policy[state])
            state, reward, terminated, info, _ = env.step(new_action)
            total_reward = reward if total_reward is Ellipsis else reward + total_reward
            env.render()

        print(f'Discount: {discount} : Reward: {total_reward}')
        if max_reward is Ellipsis:
            max_reward = total_reward
            best_discount = discount
        elif max_reward < total_reward:
            max_reward = total_reward
            best_discount = discount

    best_policy, _ = policy_iteration(env, env.action_space.n, env.observation_space.n, discount_factor=best_discount)

    print("Average steps and average reward per iterations:")
    for iterations in [50, 100]:
        total_steps = ...
        total_reward = ...
        for i in range(0, iterations):
            env.reset()
            steps = ...
            reward_current = ...
            terminated = False
            while not terminated:
                new_action = np.argmax(best_policy[state])
                state, reward, terminated, info, _ = env.step(new_action)
                reward_current = reward if reward_current is Ellipsis else reward + reward_current
                steps = 1 if steps is Ellipsis else steps + 1
                env.render()
            total_steps = steps if total_steps is Ellipsis else steps + total_steps
            total_reward = reward_current if total_reward is Ellipsis else reward_current + total_reward
        print(
            f'Iterations: {iterations} average steps: {total_steps / iterations}, average reward: {total_reward / iterations}')

# as a result i got the following output:
#
# Total reward per discount:
# Discount: 0.5 : Reward: 10
# Discount: 0.7 : Reward: 3
# Discount: 0.9 : Reward: 9
# Average steps and average reward per iterations:
# Iterations 50: avg steps: 14.5, avg reward: -2.5
# Iterations 100: avg steps: 14.26, avg reward: -2.17
#
# which means that the function is prioritizing short-term rewards from the present

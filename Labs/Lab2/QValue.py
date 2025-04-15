from q_learning import *
from numpy import *


def get_discrete_state(state, low_value, window_size):
    new_state = (state - low_value) / window_size
    return tuple(new_state.astype(np.int64))


def q_value_iterator(env, discount_factors, learning_rates, episodes, best, iterations, steps_per_episode, epsilon=None,
                     decay=None, observation_space_size=None):
    if observation_space_size is not None:
        num_actions = env.action_space.n
        observation_space_low_value = env.observation_space.low
        observation_space_high_value = env.observation_space.high
        observation_window_size = (observation_space_high_value - observation_space_low_value) / observation_space_size
    else:
        num_states = env.observation_space.n
        num_actions = env.action_space.n

    original_epsilon = epsilon

    for episode in episodes:
        for step in steps_per_episode:
            for learning_rate in learning_rates:
                for discount_factor in discount_factors:
                    if observation_space_size is not None:
                        q_table = random_q_table(-1, 0, (observation_space_size + [num_actions]))
                    else:
                        q_table = random_q_table(-1, 0, (num_states, num_actions))
                    print()
                    print(
                        f'Learning rate: {learning_rate}, Discount factor: {discount_factor}, Episodes: {episode}, Steps per Episode: {step}:')
                    for ep in range(episode):
                        state, _ = env.reset()

                        if observation_space_size is not None:
                            state = get_discrete_state(state, observation_space_low_value, observation_window_size)

                        for st in range(step):

                            if epsilon is not None:
                                action = get_action(env, q_table, state, epsilon)
                            else:
                                if best:
                                    action = get_best_action(q_table, state)
                                else:
                                    action = get_random_action(env)
                            new_state, reward, terminated, _, _ = env.step(action)

                            if observation_space_size is not None:
                                new_state = get_discrete_state(new_state, observation_space_low_value,
                                                               observation_window_size)

                            new_q = calculate_new_q_value(q_table, state, new_state, action, reward, learning_rate,
                                                          discount_factor)
                            q_table[state, action] = new_q
                            state = new_state
                            if terminated:
                                break

                        if decay is not None and epsilon is not None:  # decay posle sekoja epizoda
                            epsilon *= decay
                            if epsilon <= 0:
                                epsilon = 0.01

                    for iteration in iterations:
                        total_steps_to_goal = []
                        total_steps = []
                        total_reward_to_goal = []
                        total_reward = []
                        for i in range(iteration):
                            state, _ = env.reset()

                            if observation_space_size is not None:
                                state = get_discrete_state(state, observation_space_low_value, observation_window_size)

                            terminated = False
                            steps = 0
                            reward_current = 0
                            goal = False
                            env.render()
                            while not terminated:
                                if epsilon is not None:
                                    action = get_action(env, q_table, state, original_epsilon)
                                else:
                                    action = get_best_action(q_table, state)
                                new_state, reward, terminated, _, _ = env.step(action)

                                if observation_space_size is not None:
                                    new_state = get_discrete_state(new_state, observation_space_low_value,
                                                                   observation_window_size)

                                if reward > 0:
                                    goal = True
                                reward_current += reward
                                steps += 1
                                if steps > 1000:  # za da ne cekame beskonecno
                                    break
                                env.render()

                            total_steps.append(steps)
                            total_reward.append(reward_current)
                            if goal:  # ako stigne do celta da se ispitaat cekorite i nagradite
                                total_steps_to_goal.append(steps)
                                total_reward_to_goal.append(reward_current)

                        if len(total_steps_to_goal) > 0:
                            print(
                                f'Iterations: {iteration}:\nAverage Steps: {average(total_steps)}, Average Steps to Goal: {average(total_steps_to_goal)}\nAverage Reward: {average(total_reward)}, Average Reward to Goal: {average(total_reward_to_goal)}')
                        else:
                            print(
                                f'Iterations: {iteration}:\nAverage Steps {average(total_steps)}, Average Steps to Goal: / (Did not reach the Goal)\nAverage Reward: {average(total_reward)}, Average Reward to Goal: / (Did not reach the Goal)')

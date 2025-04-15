import gymnasium as gym

from QValue import q_value_iterator

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode='ansi')

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    learning_rates = [0.1, 0.01]
    discount_factors = [0.5, 0.9]
    episodes = [5000, 10000]
    iterations = [50, 100]
    steps_per_episode = [1000, 2000]
    epsilon = 0.5

    print("---------------------------------------------------")
    print("BEST ACTION, WITHOUT E-greedy")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, True, iterations, steps_per_episode)

    print("---------------------------------------------------")
    print(f"BEST ACTION, WITH E-greedy: {epsilon}")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, True, iterations, steps_per_episode, epsilon)

    print("---------------------------------------------------")
    print("RANDOM ACTION, WITHOUT E-greedy")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, False, iterations, steps_per_episode)

    print("---------------------------------------------------")
    print(f"RANDOM ACTION, WITH E-greedy: {epsilon}")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, False, iterations, steps_per_episode, epsilon)

# best action, without E-greedy:
# Agentot voopsto ne stigna do goal, maksimum broj na cekori, nagrada 0, mnogu losi rezultati
# best action, with E-greedy:
# Agentot stigna do goal vo sekoj test, prosecon broj na cekori so ostvarena cel okolu 450, prosecen reward so ostvarena cel okolu -2000, mnogu dobri rezultati
# random action, without E-greedy:
# Agentot voopsto ne stigna do goal, maksimum broj na cekori, nagrada 0, mnogu losi rezultati
# random action, with E-greedy:
# Agentot stigna do goal vo sekoj test, prosecon broj na cekori so ostvarena cel okolu 450, prosecen reward so ostvarena cel okolu -1500, najdobri rezultati

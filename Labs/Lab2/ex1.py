import gymnasium as gym

from QValue import q_value_iterator

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='ansi')

    discount_factors = [0.5, 0.9]
    learning_rates = [0.1, 0.01]
    episodes = [5000, 10000]
    iterations = [50, 100]
    steps_per_episode = [1000, 2000]
    epsilon = 0.5

    print("---------------------------------------------------")
    print("RANDOM ACTION, WITHOUT E-greedy")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, False, iterations, steps_per_episode)

    print("---------------------------------------------------")
    print(f"RANDOM ACTION, WITH E-greedy: {epsilon}")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, False, iterations, steps_per_episode, epsilon)

# random action, without E-greedy:
# Agentot vo globala ne stigna do goal, vo globala neshto pod maksimumot broj na cekori, nagrada okolu 0, mnogu losi rezultati
# random action, with E-greedy:
# Agentot stigna do goal vo pola od testovite, prosecon broj na cekori so ostvarena cel okolu 10, prosecen reward so ostvarena cel 1, podobri rezultati

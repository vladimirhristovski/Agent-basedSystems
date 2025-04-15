import gymnasium as gym
from QValue import q_value_iterator

if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode='rgb_array')

    decay = 0.95
    discount_factors = [0.9]
    learning_rates = [0.1]
    episodes = [10000]
    iterations = [50, 100]
    steps_per_episode = [200]
    observation_space_size = [30, 30]
    epsilon = 0.5

    print("---------------------------------------------------")
    print("WITHOUT E-decay")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, True, iterations, steps_per_episode, epsilon,
                     observation_space_size=observation_space_size)

    print("---------------------------------------------------")
    print(f"WITH E-decay: {decay}")
    print("---------------------------------------------------")

    q_value_iterator(env, discount_factors, learning_rates, episodes, True, iterations, steps_per_episode, epsilon,
                     decay, observation_space_size)

# without E-decay:
# Agentot voopsto ne stigna do goal, maksimum broj na cekori, nagrada 0, mnogu losi rezultati
# with E-decay :
# Agentot voopsto ne stigna do goal, maksimum broj na cekori, nagrada 0, mnogu losi rezultati

import numpy as np
# gym is a package that allow to play games generating an environment
import gym
import random

# Selecting the game
env = gym.make("Taxi-v2")
# Show the game
env.render()
# Action size and states size for creating the QTable
action_size = env.action_space.n
state_size = env.observation_space.n
print("Action size: " + str(action_size) + "\nState size: " + str(state_size))

qtable = np.zeros((state_size, action_size))
print(qtable)

# Hyperparameters

total_episodes      = 2000
total_test_episodes = 100
max_steps           = 79  # Max steps per episode

learning_rate       = 0.5
gamma               = 0.818  # Discounting rate

epsilon             = 1.0  # Exploration rate
max_epsilon         = 1.0  # Exploration probability at start
min_epsilon         = 0.01  # Exploration probablity at ending
# Exponential decay rate for the exploration probability
decay_rate          = 0.01

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        exp_tradeoff = random.uniform(0, 1)
        # Explotation vs exploration
        if exp_tradeoff > epsilon: # Explotation
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample() # Exploration

        new_state, reward, done, info = env.step(action)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * \
                                (
                                    reward
                                    + gamma * np.max(qtable[new_state, :]) \
                                    - qtable[state, action]
                                )
        state = new_state
        if done == True:
            break

    epsilon = min_epsilon \
            +(max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

env.reset()
rewards = []

for episode in range(total_test_episodes):
    state         = env.reset()
    step          = 0
    done          = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)
    print("****************************************************")

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        env.render()
        # Take the action (index) that have the maximum expected future reward
        # given that state
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            #print ("Score", total_rewards)
            break
        state = new_state
env.close()
print ("Score over time: " + str(sum(rewards) / total_test_episodes))

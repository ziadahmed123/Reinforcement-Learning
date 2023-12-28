import gym
import random
import numpy as np


# Discritize observation and action space in bins.
pos_space = np.linspace(-1.2, 0.6, 12)
vel_space = np.linspace(-0.07, 0.07, 20)

# given observation, returns what bin
def getState(observation):
    pos = observation[0]
    vel = observation[1]
    pos_bin = np.digitize(pos, pos_space)
    vel_bin = np.digitize(vel, vel_space)

    return (pos_bin, vel_bin)


# Creates a new empty Q-table for this environment
def createEmptyQTable():
    states = []
    for pos in range(len(pos_space) + 1):
        for vel in range(len(vel_space) + 1):
            states.append((pos, vel))
    Q = {}
    for state in states:
        for action in range(env.action_space.n):
            Q[state, action] = 0
    return Q


# Given a state and a set of actions
# returns action that has the highest Q-value
def maxAction(Q, state, actions=[0, 1, 2]):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return action

# Define the Enviroment
env = gym.make('MountainCar-v0')
env._max_episode_steps = 200
max_steps=200


def QLearning(env, episodes=5000, gamma=0.95, alpha=0.1, epsilon=1):
    # Create an empty Q-table
    Q = createEmptyQTable()

    # Keep track of total score obtained at each episode
    total_score = np.zeros(episodes)
    score = 0
    # Loop through episodes
    for i in range(episodes):
        # Print episode number every episode

        print(f'episode: {i}, total score: {score}, epsilon: {epsilon:0.3f}')

        # Reset environment and get initial observation
        observation = env.reset()
        state = getState(observation[0])

        # Loop until episode ends
        done = False
        score = 0
        steps = 0
        while not done and steps < max_steps:
            # Epsilon-greedy strategy for selecting actions
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = maxAction(Q, state)

            # Take action and observe next state and reward
            next_observation, reward, done, ter, info = env.step(action)
            next_state = getState(next_observation)

            # Update Q value for current state and action
            Q[state, action] = Q[state, action] + alpha * (
                        reward + gamma * Q[next_state, maxAction(Q, next_state)] - Q[state, action])

            # Update current state and score
            state = next_state
            score += reward
            steps += 1

        # Decay epsilon
        epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01

    return Q


QLearning_Q = {}
QLearning_Q = QLearning(env)


print("Q_Table of Q_Learning")
print(QLearning_Q)
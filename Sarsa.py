import gymnasium as gym
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

def Sarsa(env, episodes=5000, gamma=0.95, alpha=0.1, epsilon=1):
    # Create an empty Q-table
    Q = createEmptyQTable()

    score = 0
    # Variable to keep track of the total score obtained at each episode
    total_score = np.zeros(episodes)
    for i in range(episodes):
        # Keep track of total score obtained at each episode
        print(f'episode: {i}, score: {score}, epsilon: {epsilon:0.3f}')

        observation = env.reset()
        state = getState(observation[0])

        # e-Greedy strategy
        # Explore random action with probability epsilon
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        # Take best action with probability 1-epsilon
        else:
            action = maxAction(Q, state)
        score = 0
        done = False
        steps = 0
        while not done and steps < max_steps:
            # Take action and observe next state
            next_observation, reward, done, ter,info  = env.step(action)

            next_state = getState(next_observation)
            # Get next action following e-Greedy policy
            if random.uniform(0, 1) < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = maxAction(Q, next_state)
            # Add reward to the score of the episode
            score += reward
            # Update Q value for state and action given the bellman equation
            Q[state, action] = Q[state, action] + alpha * (
                        reward + gamma * Q[next_state, next_action] - Q[state, action])

            # Move to next state, and next action
            state, action = next_state, next_action
            steps += 1
        total_score[i] = score
        epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01

    return Q


Sarsa_Q = {}
Sarsa_Q = Sarsa(env)


print("Q_Table of Sarsa")
print(Sarsa_Q)
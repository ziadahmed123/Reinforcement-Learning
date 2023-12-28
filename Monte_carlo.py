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


def MonteCarlo(env, episodes=5000, gamma=0.95, alpha=0.1, epsilon=1):
    # Create an empty Q-table
    Q = createEmptyQTable()

    # Keep track of total score obtained at each episode
    total_score = np.zeros(episodes)

    # Loop through episodes
    for i in range(episodes):

        # Reset episode variables
        episode_states = []
        episode_actions = []
        episode_rewards = []

        # Reset environment and get initial observation
        observation = env.reset()
        state = getState(observation[0])

        # Epsilon-greedy strategy for selecting actions
        done = False
        steps = 0
        while not done and steps < max_steps:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = maxAction(Q, state)

            # Take action and observe next state and reward
            next_observation, reward, done, ter, info = env.step(action)
            next_state = getState(next_observation)

            # Add current state, action, and reward to episode
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            if done:
                # Update total score for episode
                total_score[i] = sum(episode_rewards)
                # Update Q-values for all state-action pairs visited in the episode
                for j in range(len(episode_states)):
                    state_j = episode_states[j]
                    action_j = episode_actions[j]
                    G_j = sum(episode_rewards[j:])  # Total reward from state j
                    Q[state_j, action_j] = Q[state_j, action_j] + alpha * (G_j - Q[state_j, action_j])

                break
            # Update current state
            state = next_state
            steps += 1
        total_score[i] = sum(episode_rewards)
        # Print episode number every episode
        print(f'episode: {i}, total score: {total_score[i]}, epsilon: {epsilon:0.3f}')
        epsilon = epsilon - 2 / episodes if epsilon > 0.01 else 0.01
    return Q


MonteCarlo_Q = {}
MonteCarlo_Q = MonteCarlo(env)


print("Q_Table of Monte Carlo")
print(MonteCarlo_Q)
#!/usr/bin/env python

# Implementation of a Markov Decision Process (MDP)

import gym
import numpy as np

# Init Environment
env = gym.make("FrozenLake-v0")


class MarkovDecisionProcess(object):
    def __init__(self, env, gamma = 0.9):
        self.gamma = gamma
        #This code would be modified to make more generalized
        self.actions_n = env.nA
        self.states_n = env.nS
        self.R = np.zeros([self.states_n, self.actions_n, self.states_n])
        self.T = np.zeros([self.states_n, self.actions_n, self.states_n])
        for s in range(self.states_n):
            for a in range(self.actions_n):
                t = env.P[s][a]
                for prob_trans, next_state, reward, done in t:
                    self.T[s,a,next_state] += prob_trans
                    if done and reward == 0.0:
                        reward += -1
                    self.R[s,a,next_state] = reward

                #Normalize T
                self.T[s,a,:]/=np.sum(self.T[s,a,:])
        # Init with a random policy
        self.policy = (1.0/self.actions_n)*np.ones([self.states_n,self.actions_n])

    def vectorize_policy(self):
        tmp_policy = np.zeros([self.states_n, self.actions_n])
        for s in range(self.states_n):
            tmp_policy[s,self.policy[s]] = 1.0
        self.policy = tmp_policy

    def calculate_mean_R(self):
        if(len(self.policy.shape) == 1):
            self.vectorize_policy()
        # Don't know how einsum works? Read this: http://ajcr.net/Basic-guide-to-einsum/
        self.R_mean = np.einsum('ijk,ijk,ij ->i', self.R, self.T, self.policy)

    def calculate_mean_T(self):
        if(len(self.policy.shape) == 1):
            self.vectorize_policy()
        self.T_mean = np.einsum('ijk,ij->ik', self.T, self.policy)

    def value_iteration(self, k=1000, gamma=1.0):
        # Calculate R_mean and T_Mean
        self.calculate_mean_R()
        self.calculate_mean_T()
        # Init Value Function to zero
        self.value_function = np.zeros(self.R_mean.shape)
        for i in range(k):
            self.value_function = self.R_mean + gamma * np.dot(self.T_mean, self.value_function)
        #print(gamma * self.value_function[None,None,:])

    def policy_iteration(self, max_iter=100, k=100, gamma=1.0):
        for __ in range(max_iter):
            optimal = self.policy.copy
            self.value_iteration(k, gamma)
            # calculate q-function
            self.q = np.einsum('ijk,ijk->ij', self.T, self.R + gamma * self.value_function[None,None,:])
            self.policy = np.argmax(self.q, axis=1)
            if np.array_equal(self.policy,optimal):
                break

def main():
    mdp = MarkovDecisionProcess(env)
    mdp.policy_iteration(max_iter=100,k=5000, gamma=0.9)
    #test optimal policy
    max_time_steps = 100000
    n_episode = 100
    env.monitor.start('logs/frozenlake-experiment', force=True)

    for i_episode in range(n_episode):
        observation = env.reset() #reset environment to beginning 
        #run for several time-steps
        for t in range(max_time_steps): 
            #sample a random action
            action = mdp.policy[observation]
            #observe next step and get reward 
            observation, reward, done, info = env.step(action)
            if done:
                env.render()
                break
    env.monitor.close()



if __name__ == '__main__':
    main()
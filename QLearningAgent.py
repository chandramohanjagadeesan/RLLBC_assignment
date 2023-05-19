import random

import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 1  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        Q_values = [self.getQValue(state, action) for action in self.actionFunction(state)]
        if len(Q_values) < 1:
            return 0
        else:
            return max(Q_values)
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if (state, action) in self.Q:
            return self.Q[(state, action)]
        else:
            return 0
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        Q_values = {}
        for a in self.actionFunction(state):
            Q_values[a] = self.getQValue(state, a)
        if len(Q_values) < 1:
            return self.getRandomAction(state)
        else:
            value = self.getValue(state)
            # gives all the actions which have the best value
            actions = [action for action in self.actionFunction(state) if self.getQValue(state, action) == value]
            if len(actions) > 1:
                return random.choice(actions)
            else:
                return actions[0]

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return None

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        Q_values = [self.getQValue(state, action) for action in self.actionFunction(state)]
        if len(Q_values) < 1:
            return self.getRandomAction(state)
        else:
            if np.random.rand() < self.epsilon:
                return self.getRandomAction(state)
            else:
                return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        if (state, action) in self.Q:
            self.Q[(state, action)] = self.getQValue(state, action) + (self.learningRate * (reward + (self.getValue(nextState) * self.discount) -
                                                            self.getQValue(state, action)))
        else:
            self.Q[(state, action)] = 0
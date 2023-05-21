from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        self.V = {}
        for s in states:
            self.V[s] = 0

        # ************

        for i in range(iterations):
            newV = {}
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                if len(actions) == 0:
                    newV[s] = 0
                else:
                    maximum = -np.inf
                    for a in actions:
                        qValue = self.getQValue(state=s, action=a)
                        maximum = max(maximum, qValue)
                    
                    newV[s] = maximum

            # Update value function with new estimate
            self.V = newV
            # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        Qvalue = 0
        trans_state_prob = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, probability in trans_state_prob:
            Qvalue += probability*(self.mdp.getReward(state, action, next_state)+(self.discount*self.getValue(state=next_state)))
        
        return Qvalue
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:
            # **********
            # TODO 2.4
            qValues = []
            for a in actions:
                qValues.append(self.getQValue(state=state, action=a))
            # getting the index that maximizes the q-value
            for i, q in enumerate(qValues):
                if q == max(qValues):
                    index = i
                    
            return actions[index]
            # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass

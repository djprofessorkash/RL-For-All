"""
This file covers the implementation of the Policy Gradient
algorithm for reinforcement learning.

The major two architectures designed cover two distinct
mathematical policies: logistic and softmax.
"""

import numpy as np

class PolicyGradient(object):

    def __init__(self, policy, parameters, learning_rate, discount_factor)
        self.policy = policy        # Policy can be logistic-based or softmax-driven
        self.θ = parameters         # Theta (θ) often represents model parameters
        self.α = learning_rate      # Alpha (α) often represents learning rate
        self.γ = discount_factor    # Gamma (γ) often represents discount factor
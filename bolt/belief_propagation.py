from functools import reduce
from collections import OrderedDict
import numpy as np

def sum_states(m, neighbors, states, function):
    neighbors = neighbors.copy()
    neighbor = neighbors[0]
    neighbors.remove(neighbor)
    if neighbors:
        for state in neighbor.states:
            states[neighbor] = state
            m = m + sum_states(m, neighbors, states, function)
    else:
        m = function(states) * 
    return m

class FNode:

    def __init__(self, function, states, neighbors):
        self.function = function
        self.states = states
        self.neighbors = neighbors
        self.is_leaf = len(neighbors) == 1
        self.messages = {}
        for n in self.neighbors:
            self.messages[n] = {}
            for self.neighbors
    
    def send(self, target):

        def sum_states(m, neighbors, states):
            neighbors = neighbors.copy()
            neighbor = neighbors[0]
            neighbors.remove(neighbor)
            if neighbors:
                for state in neighbor.states:
                    states[neighbor] = state
                    m = m + sum_states(m, neighbors, states)
            else:
                messages_from_x = [n.message[states[n]] if n != target else 1 for n in self.neighbors]
                m = self.function(states) * reduce(lambda x, y: x * y, messages_from_x)
            return m

        neighbors = self.neighbors.copy()
        neighbors.remove(target)
        for target_state in target.states:
            for n in self.neighbors:
                states[n] = None
            messages = [n.message[states[n]] if n != target else 1 for n in self.neighbors]
            
            message = self.function() * reduce(lambda x, y: x * y, messages)
            target.recieve(self, message)

    def recieve(self, neighbors, state, message):
        self.messages[state]
    
class XNode:

    def __init__(self, states, neightbors):
        self.states = states
        self.neightbors = neightbors
        self.messages = {}
        for state in self.states:
            self.messages[state] = None
        self.is_leaf = len(neighbors) == 1
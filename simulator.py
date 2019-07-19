import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Runner:
    def __init__(self):
        pass

class Agent:

    dimensions = ['d1', 'd2', 'd3', 'd4']

    @staticmethod
    def generate_profile():
        scores = np.random.rand(len(Agent.dimensions)) * 2 - 1
        return dict(zip(Agent.dimensions, scores))

    @staticmethod
    def distance(population, a, b):
        pass

    @staticmethod
    def select(population, a=None):
        pass

    @staticmethod
    def interact(population, a, b):
        pass
 
class Population:

    def __init__(self, size, topology='none', agentclass=Agent):
        if topology == 'none':
            graph = nx.empty_graph(size)
        elif topology == 'smallworld':
            graph = nx.barabasi_albert_graph(size, int(np.log(size)))

        agents = [agentclass.generate_profile() for x in range(size)]
        attributes = dict(zip(range(size), agents))
        nx.set_node_attributes(graph, attributes)

        self._size = size
        self._graph = graph
        self._agentclass = agentclass

    @property
    def size(self):
        return self._size

    @property
    def graph(self):
        return self._graph


if __name__ == "__main__":
    p = Population(10, 'smallworld')
    nx.draw(p.graph)
    plt.show()

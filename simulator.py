import copy
import numpy as np
import networkx as nx

import dask.threaded
from dask import compute, delayed

class Runner:
    def __init__(self, population, nsim, snapshot_rate=10):
        population_size = population.size
        self._p = [copy.deepcopy(population) for x in range(nsim)]
        self._snapshot_rate = snapshot_rate

    def _evolve(self, population, nsteps):
        _ = [population.tic() for x in range(nsteps)]

    def _epoch(self, nsteps):
        tasks = [delayed(self._evolve)(p, nsteps) for p in self._p]
        compute(*tasks, scheduler='threads')

    def _take_snapshot(self):
        print('click!')

    def run(self, nsteps=1000):
        nepochs = nsteps // self._snapshot_rate
        odds = nsteps % self._snapshot_rate
        
        for i in range(nepochs):
            self._epoch(self._snapshot_rate)
            self._take_snapshot()

        if odds > 0:
            self._epoch(odds)
            self._take_snapshot()


class Agent:

    dimensions = ['d1', 'd2', 'd3', 'd4', 'd5']

    @staticmethod
    def generate_profile():
        scores = np.random.rand(len(Agent.dimensions)) * 2 - 1
        return dict(zip(Agent.dimensions, scores))

    @staticmethod
    def distance(population, a, b):
        # euclidean distance * sine similarity
        v_a = np.array([population.nodes[a][d] for d in Agent.dimensions])
        v_b = np.array([population.nodes[b][d] for d in Agent.dimensions])

        d = np.sqrt(np.sum([a ** 2 + b ** 2 for a, b in zip(v_a, v_b)]))
        theta = v_a.dot(v_b) / np.sqrt(np.sum(v_a ** 2) * np.sum(v_b ** 2))
        return d * np.sin(theta)


    @staticmethod
    def select(population, a=None):
        if a == None:
            selected = np.random.choice(population.nodes)
        else:
            selected = a
            while selected == a:
                selected = np.random.choice(population.nodes)

        return selected

    @staticmethod
    def interact(population, a, b):
        dimension = np.random.choice(Agent.dimensions)
        node_a = population.nodes[a]
        node_b = population.nodes[b]
        weighted_mean = np.mean((node_a[dimension], node_b[dimension])) * 0.01
        node_a[dimension] += weighted_mean
        node_b[dimension] += weighted_mean

    def set_relation(population, a, b):
        dist = Agent.distance(population, a, b)
        if dist > 0:
            population.add_edge(a, b, weight=dist)
        elif population.has_edge(a,b):
            population.add_edge(a, b, weight=0)
 
class Population:

    def __init__(self, size, topology='none', agentclass=Agent):
        if topology == 'none':
            graph = nx.empty_graph(size)
        elif topology == 'smallworld':
            graph = nx.barabasi_albert_graph(size, int(np.log(size)))

        agents = [agentclass.generate_profile() for x in range(size)]
        attributes = dict(zip(range(size), agents))
        nx.set_node_attributes(graph, attributes)

        _ = [Agent.set_relation(graph, *edge) for edge in graph.edges]

        self._size = size
        self._graph = graph
        self._agentclass = agentclass

    def tic(self):
        emitter_node = Agent.select(self._graph)
        receiver_node = Agent.select(self._graph, emitter_node)

        Agent.interact(self._graph, emitter_node, receiver_node)
        Agent.set_relation(self._graph, emitter_node, receiver_node)

    @property
    def size(self):
        return self._size

    @property
    def graph(self):
        return self._graph


if __name__ == "__main__":
    pop_size = 1000
    n_pops = 10
    sr = 10
    steps = 1000

    import datetime

    print('[', datetime.datetime.now().time(), ']')
    print(f"Initializing: population_size [{pop_size}] n_populations [{n_pops}]")

    p = Population(pop_size, 'smallworld')
    sim = Runner(p, n_pops, sr)

    print('[', datetime.datetime.now().time(), ']')
    print("Allocation of the resources completed!")

    sim.run(100)

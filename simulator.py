import copy
import pickledb
import numpy as np
import networkx as nx

import dask.threaded
from dask import compute, delayed

class Runner:
    def __init__(self, population, nsim, outfilename, snapshot_rate=10):
        self._nsim = nsim
        self._p = [copy.deepcopy(population) for x in range(nsim)]
        self._snapshot_rate = snapshot_rate
        self._counter = 0
        self._outdb = self._init_outdb(outfilename)
        self._take_snapshot()

    def _init_outdb(self, outfilename):
        outdb = pickledb.load(outfilename, False)
        params = {
                'nsim': self._nsim,
                'snapshot_rate': self._snapshot_rate
        }
        outdb.set('_params', params)
        outdb.dump()

        return outdb

    def _evolve(self, population, nsteps):
        _ = [population.tic() for x in range(nsteps)]

    def _epoch(self, nsteps):
        tasks = [delayed(self._evolve)(p, nsteps) for p in self._p]
        compute(*tasks, scheduler='threads')

    def _serialize(self, population):
        return ''.join(nx.generate_gml(population.graph))

    def _take_snapshot(self):
        tasks = [delayed(self._serialize)(p) for p in self._p]
        values = compute(*tasks, scheduler='threads')
        for i, val in enumerate(values):
            key = f'{i}_{self._counter}'
            self._outdb.set(key, val)
        self._outdb.dump()

    def run(self, nsteps=1000):
        # nsteps MUST be divisible by the sampling_rate
        # extra steps will be ignored (design choice)
        nepochs = nsteps // self._snapshot_rate
        
        for i in range(nepochs):
            self._epoch(self._snapshot_rate)
            self._counter += self._snapshot_rate
            self._take_snapshot()


class Agent:

    dimensions = ['d1', 'd2']

    @classmethod
    def generate_profile(cls):
        scores = np.random.rand(len(cls.dimensions)) * 2 - 1
        return dict(zip(cls.dimensions, scores))

    @classmethod
    def distance(cls, population, a, b):
        # euclidean distance * sine similarity
        v_a = np.array([population.nodes[a][d] for d in cls.dimensions])
        v_b = np.array([population.nodes[b][d] for d in cls.dimensions])

        d = np.sqrt(np.sum([a ** 2 + b ** 2 for a, b in zip(v_a, v_b)]))
        theta = v_a.dot(v_b) / np.sqrt(np.sum(v_a ** 2) * np.sum(v_b ** 2))
        return d * np.sin(theta)


    @classmethod
    def select(cls, population, a=None):
        if a == None:
            selected = np.random.choice(population.nodes)
        else:
            selected = a
            while selected == a:
                selected = np.random.choice(population.nodes)

        return selected

    @classmethod
    def interact(cls, population, a, b):
        dimension = np.random.choice(cls.dimensions)
        node_a = population.nodes[a]
        node_b = population.nodes[b]
        weighted_mean = np.mean((node_a[dimension], node_b[dimension])) * 0.01
        node_a[dimension] += weighted_mean
        node_b[dimension] += weighted_mean

    @classmethod
    def set_relation(cls, population, a, b):
        dist = cls.distance(population, a, b)
        if dist > 0:
            population.add_edge(a, b, weight=dist)
        elif population.has_edge(a,b):
            population.add_edge(a, b, weight=0)
 
class Population:

    def __init__(self, size, topology='none', agentcls=Agent):
        if topology == 'none':
            graph = nx.empty_graph(size)
        elif topology == 'smallworld':
            graph = nx.barabasi_albert_graph(size, int(np.log(size)))

        agents = [agentcls.generate_profile() for x in range(size)]
        attributes = dict(zip(range(size), agents))
        nx.set_node_attributes(graph, attributes)

        _ = [agentcls.set_relation(graph, *edge) for edge in graph.edges]

        self._size = size
        self._graph = graph
        self._agent = agentcls

    def tic(self):
        emitter_node = self._agent.select(self._graph)
        receiver_node = self._agent.select(self._graph, emitter_node)

        self._agent.interact(self._graph, emitter_node, receiver_node)
        self._agent.set_relation(self._graph, emitter_node, receiver_node)

    @property
    def size(self):
        return self._size

    @property
    def graph(self):
        return self._graph


if __name__ == "__main__":
    pop_size = 100
    n_pops = 1
    sr = 10
    steps = 1000

    import datetime

    print('[', datetime.datetime.now().time(), ']')
    print(f"Initializing: population_size [{pop_size}] n_populations [{n_pops}]")

    p = Population(pop_size, 'smallworld')
    sim = Runner(p, n_pops, 'sim.outdb', sr)

    print('[', datetime.datetime.now().time(), ']')
    print("Starting evolution of the populations..")

    sim.run(steps)

    print('[', datetime.datetime.now().time(), ']')
    print("Completed!")

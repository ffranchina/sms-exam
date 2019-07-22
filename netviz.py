import os
import os.path
import shutil
import tempfile
import pickledb
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


class Helper:
    
    def __init__(self, G):
        self._g = G
        self._dimensions = G.nodes['0'].keys()

    def color(self, node):
        vec = np.array([self._g.nodes[node][d] for d in self._dimensions])
        ref = np.zeros_like(vec)
        ref[0] = 1
        
        d = np.sqrt(np.sum([e ** 2 for e in vec]))
        theta = vec.dot(ref) / np.sqrt(np.sum(vec ** 2) * np.sum(ref ** 2))
        return d * np.sin(theta)

    def weights_color(self):
        weights = [attr['weight'] for *_, attr in self._g.edges(data=True)]
        weights = np.array(weights)
        weights = weights / np.max(weights)
        colors = [(0.2, 0.2, 0.2, w) for w in weights]
        return colors

    @property
    def dimensions(self):
        return self.dimensions


class RandomGraph:

    def __init__(self, filename):
        self._outdb = pickledb.load(filename, False)
        tmpgraph = nx.parse_gml(self._outdb.get('0_0'))
        self._pos = nx.random_layout(tmpgraph)

        nkeys = self._outdb.totalkeys() - 1
        snapshot_rate = self._outdb.get('_params')['snapshot_rate']
        self._snapshots = range(0, nkeys * snapshot_rate, snapshot_rate)

    def _load(self, graphid):
        return nx.parse_gml(self._outdb.get(graphid))

    def _plot(self, graphid):
        G = self._load(graphid) 
        helper = Helper(G)
        nodecolors = [helper.color(node) for node in G.nodes]
        nodesize = 30
        edgecolor = helper.weights_color()

        nodes = nx.draw_networkx_nodes(G, self._pos, \
                node_color=nodecolors, node_size=nodesize, linewidths=.3, \
                edgecolors='#333333', cmap=plt.cm.Spectral)
        edges = nx.draw_networkx_edges(G, self._pos, node_size=nodesize, \
                width=.5)
        edges._edgecolors = mcolors.to_rgba_array(edgecolor)

    def plot_gif(self, trackid, filename, deleteimgs=True):
        tmpdir = tempfile.mkdtemp()
        imgcounter = 0
        for i in self._snapshots:
            graphid = f'{trackid}_{i}'
            imgname = '%04d.png' % imgcounter
            path = os.path.join(tmpdir, imgname)
            self.plot(graphid, path)
            imgcounter += 1

        os.system(f'convert {tmpdir}/* -delay 50 -loop 0 {filename}')
        
        if deleteimgs:
            shutil.rmtree(tmpdir)

    def plot(self, graphid, filename):
        self._plot(graphid)
        plt.savefig(filename, dpi=150, bbox_inches='tight', \
                pad_inches=-0.05)
        plt.clf()


r = RandomGraph('sim.outdb')
#r.plot('0_0', 'prova.png')
r.plot_gif('0', 'prova.gif', False)

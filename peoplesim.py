import argparse

import simulator
import netviz

parser = argparse.ArgumentParser(
        description="Population interaction - Simulator and Analysis")
subparsers = parser.add_subparsers(
        dest='command', help='Action to perform')
subparsers.required = True

parser_generate = subparsers.add_parser('generate',
        help='Generate interaction trajectories')

parser_generate.add_argument('-n', '--npopulations', action='store', 
        default=10, type=int, help='Number of populations to simulate')
parser_generate.add_argument('-p', '--psize', action='store', default=100,
        type=int, help='Size of the population to simulate')
agent_types = [attr for attr in dir(simulator) if attr.endswith('Agent')]
parser_generate.add_argument('agent', action='store', choices=agent_types,
        help='Determines the interaction policies between individuals')
topologies = ('none', 'erdosrenyi', 'smallworld')
parser_generate.add_argument('-t', '--topology', action='store',
        default='none', choices=topologies, help='Initial social topology')
parser_generate.add_argument('-s', '--steps', action='store', default=1000,
        type=int, help='Number of the steps performed by simulation')
parser_generate.add_argument('-r', '--rate', action='store', default=10,
        type=int, help='Rate with which snapshots will be taken')
parser_generate.add_argument('-o', '--output', action='store',
        default='sim.outdb', help='Output file name')

parser_visualize = subparsers.add_parser('visualize',
        help='Visualize the data coming from a simulation')

plot_types = [attr for attr in dir(netviz) if attr.endswith('Graph')]
parser_visualize.add_argument('type', action='store', choices=plot_types,
        help='Type of plot to output')
parser_visualize.add_argument('-i', '--input', action='store',
        default='sim.outdb', help='Output image file name')
parser_visualize.add_argument('target', action='store',
        help='Track ID and step ID (es. 1_*)')
parser_visualize.add_argument('output', action='store',
        help='Output image file name')


def generate(n_populations, population_size, agent, topology, steps,
        snapshot_rate, output_filename):
    agent_class = getattr(simulator, agent)

    p = simulator.Population(population_size, agent_class, topology)
    sim = simulator.Runner(p, n_populations, output_filename, snapshot_rate)

    sim.run(steps)

def visualize(plot_type, input_filename, graph_id, output_filename):
    track_id, step_id = graph_id.split("_")

    plotting_class = getattr(netviz, plot_type)
    plt = plotting_class(input_filename, track_id)

    if output_filename.endswith('.gif'):
        plt.plot_gif(output_filename)
    else:
        plt.plot(step_id, output_filename)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.command == 'generate':
        generate(args.npopulations, args.psize, args.agent, args.topology,
                args.steps, args.rate, args.output)
    
    elif args.command == 'visualize':
        visualize(args.type, args.input, args.target, args.output)

import argparse

import networkx as nx
import numpy as np
from graph_tool.all import *

from GFP import GFPSingleFingerprint


def load_graph_data(args):
    """Load all the graph data and return GraKal graph object"""
    
    graph_list = []
    num_graphs = 1000
    graph_types = ['BA', 'ER', 'FF', 'RM', 'SW']

    for graph_type in graph_types:
        print(f'Loading the {graph_type} graphs')
        for i in range(num_graphs):
            g = Graph()
            g.add_edge_list(np.array([line.split(",") for line in nx.generate_edgelist(nx.read_edgelist(f'{args.data_loc}/{graph_type}/{graph_type}_{i}.edges'), delimiter=',', data=False)]), hashed=True)
            g.set_directed(False)
            graph_list.append(g)

    return graph_list
    

def compute_gfp(graphs):
    """Compute all the finger prints for an input list of graphs"""

    gfp_list = []
    for graph in graphs:

        gfp_list.append(GFPSingleFingerprint(graph))
        print("Graph has been extracted")

    gfp_list = [sublist[0] + sublist[1] for sublist in gfp_list]
    gfp_list = np.array(gfp_list)
    np.save('gfps.npy', gfp_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loc', type=str, default='')  
    args = parser.parse_args()

    graph_list = load_graph_data(args)
    compute_gfp(graph_list)    

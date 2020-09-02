import numpy as np
import snap
import argparse

def gen_ff(args):
    """Generate FF Graph"""

    for i in range(args.num_graphs):

        fp = np.random.uniform(0, 0.5)
        bp = np.random.uniform(0, 0.5)
    
        Graph = snap.GenForestFire(args.num_vertices, fp, bp)
        snap.SaveEdgeList(Graph, f'{args.data_loc}/FF/FF_{i}.edges')

        print(f"FF Graph {i} Generated and Saved")

def gen_ba(args):
    """Generate a BA Graph"""

    for i in range(args.num_graphs):

        out_deg = int(np.random.uniform(2, 6))
        Rnd = snap.TRnd()
        Graph = snap.GenPrefAttach(args.num_vertices, out_deg, Rnd)
        snap.SaveEdgeList(Graph, f'{args.data_loc}/BA/BA_{i}.edges')

        print(f"BA Graph {i} Generated and Saved")

def gen_sw(args):
    """Generate a SW Graph"""

    for i in range(args.num_graphs):

        fp = np.random.uniform(0, 0.5)
        Rnd = snap.TRnd()
        Graph = snap.GenSmallWorld(args.num_vertices, 3, fp, Rnd)
        snap.SaveEdgeList(Graph, f'{args.data_loc}/SW/SW_{i}.edges')

        print(f"SW Graph {i} Generated and Saved")

def gen_rm(args):
    """Generate a RM Graph"""

    for i in range(args.num_graphs):

        a = np.random.uniform(0, 0.3)
        b = np.random.uniform(0, 0.1)
        c = np.random.uniform(0, 0.1)
        num_edges = int(np.random.uniform((args.num_vertices), (args.num_vertices*2)))
        Graph = snap.GenRMat(args.num_vertices, num_edges, a, b, c)
        snap.SaveEdgeList(Graph, f'{args.data_loc}/RM/RM_{i}.edges')

        print(f"RM Graph {i} Generated and Saved")

def gen_er(args):
    """Generate a ER Graph"""

    for i in range(args.num_graphs):

        num_edges = int(np.random.uniform((args.num_vertices/2), (args.num_vertices*2)))
        Graph = snap.GenRndGnm(snap.PNGraph, args.num_vertices, num_edges)
        snap.SaveEdgeList(Graph, f'{args.data_loc}/ER/ER_{i}.edges')

        print(f"ER Graph {i} Generated and Saved")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_vertices', type=int, default=1000)
    parser.add_argument('--num_graphs', type=int, default=1000)
    parser.add_argument('--data_loc', type=str, default='')  

    args = parser.parse_args()

    gen_ff(args)
    gen_ba(args)
    gen_sw(args)
    gen_rm(args)
    gen_er(args)
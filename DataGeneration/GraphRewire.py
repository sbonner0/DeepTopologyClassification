from graph_tool.all import *
import os, csv, sys
import numpy as np
import GFP

def randomRewrite(tempG, statModel, iterations):
    # Method to rewire the graph based on some probabaltic methods.
    # Does not increase or decrease the number of vertices or edges.
    # Will rewire the graph in place
    #https://graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.random_rewire

    print random_rewire(tempG, model = statModel, n_iter = iterations, edge_sweep = False)

    return tempG

def loadDataAndGenPKL(inputdir, filename):
    filehandler = open(filename, 'wb')
    # Load the graph data.
    for subdir, dirs, files in os.walk(inputdir):
        for filename in files:
            label = subdir.split("/")
            label =  label[len(label)-1]
            g = Graph()
            edges = []
            filepath = subdir + os.sep + filename
            date =  datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print date + ": " + filepath
            sys.stdout.flush()
            with open(filepath) as networkData:
                datareader = csv.reader(networkData, delimiter="        ")
                for row in datareader:
                    if not row[0].startswith("#"):
                        edges.append([int(row[0]), int(row[1])])
            networkData.close()
            g.add_edge_list(edges, hashed=True) # Very important to hash the values here otherwise it creates too many nodes
            g.set_directed(False)

            # Randomly Rewire The Graph To Alter Topology
            itr = np.random.randint(100, 10000, size=1)
            g = randomRewrite(g, 'erdos', itr[0])

            # Pass the graph to the single fingerprint generation method and return the fingerprint vector
            fp = GFP.GFPSingleFingerprint(g)

            # Save the label, amount rewired and the fingerprint
            res = ["RW", itr, fp]
            pickle.dump(res, filehandler)
    filehandler.close()

    return 0

if __name__ == "__main__":

    print("Loading, Rewiring and Generating Fingerprints")
    loadDataAndGenPKL(sys.argv[1],sys.argv[2])

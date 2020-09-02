from graph_tool.all import *
import os, csv, time, datetime, sys
import GFP
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def loadDataAndGenPKL(inputdir, filename):
    filehandler = open(filename, 'wb')
    # Load the graph data. Need to think about the directory structure and balance of the datasets
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
                datareader = csv.reader(networkData, delimiter="	")
                for row in datareader:
                    if not row[0].startswith("#"):
                        edges.append([int(row[0]), int(row[1])])
            networkData.close()
            g.add_edge_list(edges, hashed=True) # Very important to hash the values here otherwise it creates too many nodes
            g.set_directed(False)

    # Pass the graph to the single fingerprint generation method and return the fingerprint vector
            fp = GFP.GFPSingleFingerprint(g)
            res = [label, filename, fp]
            pickle.dump(res, filehandler)

    filehandler.close()
    return 0

def usage():
    print """Usage:\n%s </path/to/input> <pickle filename>\nNB: Piclke created @ launch location unless absloute path used""" % (sys.argv[0])

if __name__ == "__main__":

    if len (sys.argv) != 3 :
        usage()
        sys.exit (1)

    loadDataAndGenPKL(sys.argv[1],sys.argv[2])

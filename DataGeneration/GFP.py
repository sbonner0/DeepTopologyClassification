from graph_tool.all import *
import os, csv
from tabulate import tabulate
import numpy as np
from scipy import stats
import scipy.spatial.distance
from numpy.random import *
import numpy

def implicitExtraction(tempG):
    print("Starting Implicit Extraction")

    # 1) - number of neigbours of each vertex
    tempG.vertex_properties["dp"] = tempG.degree_property_map("total")

    # 2) - cluster coeffecient
    tempG.vertex_properties["lc"] = local_clustering(tempG)

    # 8) Page rank
    tempG.vertex_properties["pR"] = pagerank(tempG)

    # 9) eigenvector
    tempG.vertex_properties["eV"] = eigenvector(tempG)[1]

    return tempG

def vertexFeatureExtraction(v, tempG, egoNet):
    averageNeighbourhoodScore = 0
    twoHopAwayNeighbours = 0
    egoNet.a = False
    egoNet[v] = True

    # iterate over the neighbour vertices of the current vertex
    for w in v.out_neighbours():
        # 3) - Average number of nodes two hops away
        # This is the out degree of the first neighbour (Assuming no parallel edges)
        twoHopAwayNeighbours += tempG.vp.dp[w]

        # 4) - Average clustering coeffecient of the Neihbourhood of node i
        averageNeighbourhoodScore += tempG.vp.lc[w]

        # 5) - Ego net
        egoNet[w] = True

    # Store the computed results in the property maps of the graph
    # Fix the possibily of zero degrees
    if float(tempG.vp.dp[v]) != 0.0:
        tempG.vp.tHN[v] = ((1.0 / float(tempG.vp.dp[v])) * float(twoHopAwayNeighbours))
        tempG.vp.nCCP[v] = ((1.0 / float(tempG.vp.dp[v])) * float(averageNeighbourhoodScore))
    else:
        tempG.vp.tHN[v] = ((1.0 / 1.0) * float(twoHopAwayNeighbours))
        tempG.vp.nCCP[v] = ((1.0 / 1.0) * float(averageNeighbourhoodScore))

    return tempG

def globalFeatureExtraction(tempG):

    # Extract the global features comparing the size of the network
    numEdges = tempG.num_edges()
    numVertices = tempG.num_vertices()
    gc = global_clustering(tempG)
    comp, hist = label_components(tempG)
    numComponents = len(hist)
    d = tempG.degree_property_map("total")
    num_triangles = gc[0] * (d.a * (d.a - 1) / 2).sum() / 3
    total_hist = vertex_hist(tempG, "total")
    degree_max = total_hist[1][len(total_hist[1])-2]

    f = [numEdges, numVertices, degree_max, gc[0], numComponents, num_triangles]
    return f

def GFPFeatureExtraction(tempG):
    print("Starting Feature Extraction")

    # Extract the features implcit to graph-tool
    tempG = implicitExtraction(tempG)

    # Create the property maps to store results-------------------------------------------------------
    # 3) - Average number of nodes two hops away.....
    tempG.vertex_properties["tHN"] = tempG.new_vertex_property("double")

    # 4) - Average clustering coeffecient of the Neihbourhood of node i
    tempG.vertex_properties["nCCP"] = tempG.new_vertex_property("double")

    # 5) - Number of edges in i's egonet
    tempG.vertex_properties["nEEG"] = tempG.new_vertex_property("double")
    egoNet = tempG.new_vertex_property("bool")
    egoNet.a = False

    # 6) - Number of out going edges from the neigbourhood of node i
    tempG.vertex_properties["oEEG"] = tempG.new_vertex_property("double")

    # 7) - Number of neighbours of the egonet
    tempG.vertex_properties["oNEG"] = tempG.new_vertex_property("double")

    print("Starting Vertex Level Extraction")

    for v in tempG.vertices():
        tempG = vertexFeatureExtraction(v, tempG, egoNet)
    return tempG

def GFPFeatureCreation(tempG):
    print("Starting Feature Creation")
    # Create vertex * feature matrix
    # Loop through all the vertices and extract the vertices and attributes then all to a list
    featuresCollection = [ [], [], [], [], [], [] ]
    f = []

    for v in tempG.vertices():
        featuresCollection[0].append(tempG.vp.dp[v])
        featuresCollection[1].append(tempG.vp.lc[v])
        featuresCollection[2].append(tempG.vp.tHN[v])
        featuresCollection[3].append(tempG.vp.nCCP[v])
        featuresCollection[4].append(tempG.vp.pR[v])
        featuresCollection[5].append(tempG.vp.eV[v])

    for i in range(6):
        median = numpy.median(featuresCollection[i])
        mean = numpy.mean(featuresCollection[i])
        stdev = numpy.std(featuresCollection[i])
        skewness = stats.skew(featuresCollection[i])
        kurtosis = stats.kurtosis(featuresCollection[i])
        variance = stats.tvar(featuresCollection[i])
        maxVal = stats.tmax(featuresCollection[i])
        minVal = stats.tmin(featuresCollection[i])
        f += [median, mean, stdev, skewness, kurtosis, variance, maxVal, minVal]

    return f

def GFPCompare(f1, f2):

    return abs(scipy.spatial.distance.canberra(f1, f2))

def GFPSingleFingerprint(tempG):
    # Generate a FingerPrint for a single graph
    tempG = GFPFeatureExtraction(tempG)
    features = GFPFeatureCreation(tempG)
    gloalFeatures = globalFeatureExtraction(G1)

    return [features, gloalFeatures]

def GFPControl(G1, G2):

    tempG = GFPFeatureExtraction(G1)
    features = GFPFeatureCreation(tempG)
    tempG2 = GFPFeatureExtraction(G2)
    features2 = GFPFeatureCreation(tempG2)
    vertexComparison = GFPCompare(features, features2)

    gloalFeatures = globalFeatureExtraction(G1)
    gloalFeatures2 = globalFeatureExtraction(G2)
    globalCompare = GFPCompare(gloalFeatures, gloalFeatures2)

    return [vertexComparison, globalCompare]

if __name__ == "__main__":
    # Testing function with random networks for dev work
    print("Testing with random Barabasi networks")
    g = price_network(20000, m = 2, directed = False)
    g2 = price_network(20000, m = 2, directed = False)
    print GFPControl(g, g2)

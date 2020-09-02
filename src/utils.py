# Stephen Bonner 2016 - Durham University
# Various functions for utilities for graph classification
import numpy as np
from sklearn import preprocessing
import _pickle as cPickle
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
import itertools

def scaleFP(a):
    ''' Scale a vector or a matrix such that it has has zero mean and unit variance -- Returns ndarray '''

    scaled = preprocessing.scale(a, axis=0) # Normalise each feature independently

    return scaled

def normFP(a):
    ''' Scale a vector or a matrix such that all values are between 0 and 1 -- Returns ndarray '''

    min_max_scaler = preprocessing.MinMaxScaler()
    norm = min_max_scaler.fit_transform(a)

    return norm

def unPickleFingerPrints(filename):
    ''' Load a pickle and return it as a ndarray object '''

    f = open(filename, "rb")
    objs = []
    temp = []
    temp2 = []
    while 1:
        try:
            objs.append(cPickle.load(f, encoding='iso-8859-1'))
        except EOFError:
            break

    # Loop through the unpickled object and create array
    for i in objs:
        temp.append(i[2])
        temp2.append(i[0])

    # Convert to ndarray
    features = np.asarray(temp)
    labels = np.asarray(temp2)

    return features, labels

def splitTestTrain(data, labels, percentage):
    ''' Split data into test and train '''

    X_train, X_test, y_train, y_test = cross_validate.train_test_split(data, labels, test_size=percentage, random_state=42)

    return X_train, X_test, y_train, y_test

def encodeLabels(labels):
    '''Encode the labels using one hot encoding for neural network'''

    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)

    # convert integers to dummy variables (i.e. one hot encoded)
    labels = np_utils.to_categorical(encoded_Y)

    return labels

def encodeBinaryLabels(labels):
    '''Encode the labels using one hot encoding for neural network'''

    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)

    return labels

def load_data_npy():
    """Load the numpy data format"""

    # load the full matrix of all data
    full_data = np.load('Data/gk-compare.npy')
    scaled_full_data = scaleFP(full_data)

    # graph_types = ['BA', 'ER', 'FF', 'RM', 'SW']
    # Create the list of labels
    labels = []
    for i in range(5):
        labels.append(np.full((1, 1000), i))
    labels = np.array(labels).flatten()

    return scaled_full_data, labels, full_data

def loadData(ecLabels):
    ''' Function for loading the pickled graph fingerprints '''

    # Load Barabasi Dataset
    BAfeatures, BAlabels = unPickleFingerPrints("Data/BA.pkl")

    # Load Random Dataset
    ERfeatures, ERlabels = unPickleFingerPrints("Data/ER.pkl")

    # Load FF Dataset
    FFfeatures, FFlabels = unPickleFingerPrints("Data/FF.pkl")

    # Load R-MAT Dataset
    RMfeatures, RMlabels = unPickleFingerPrints("Data/RM.pkl")

    # Load Small World Dataset
    SWfeatures, SWlabels = unPickleFingerPrints("Data/SW.pkl")


    # Concatenate the datasets together
    unScaledFeatures = np.concatenate((BAfeatures, ERfeatures, FFfeatures[:-1], RMfeatures, SWfeatures), axis=0)
    features = scaleFP(unScaledFeatures)

    # graph_types = ['BA', 'ER', 'FF', 'RM', 'SW']
    # Create the list of labels
    labels = []
    for i in range(5):
        labels.append(np.full((1, 10000), i))
    labels = np.array(labels).flatten()

    return features, labels, unScaledFeatures

def loadAnomData(ecLabels):
    ''' Function for loading the rewired pickled graph fingerprints '''

    # Load FF Dataset
    FFfeatures, FFlabels = unPickleFingerPrints("Data/FF.pkl")
    FFscaledFeatures = scaleFP(FFfeatures)

    # Load ANOM Dataset
    ANOMfeatures, ANOMlabels = unPickleFingerPrints("Data/ANOM.pkl")
    ANOMscaledFeatures = scaleFP(ANOMfeatures)

    # Concatenate the datasets togther
    features = np.concatenate((FFscaledFeatures, ANOMscaledFeatures), axis=0)
    labels = np.concatenate((FFlabels, ANOMlabels), axis=0)
    unScaledFeatures = np.concatenate((FFfeatures, ANOMfeatures), axis=0)

    # Encode the labels using one hot encoding if required
    if ecLabels:
        labels = encodeBinaryLabels(labels)

    return features, labels, unScaledFeatures

def takeJustGlobal(features):
    ''' Take just the last six columns from every row '''

    return features[:, -6:]

def takeJustLocal(features):
    ''' Take just the local features from the feature matrix '''

    return features[:, :-6]
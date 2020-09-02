# Stephen Bonner 2016 - Durham University
# Various functions for utilites for graph classification

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import preprocessing
import cPickle
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
import itertools

def scaleFP(a):
    ''' Scale a vector or a matrix such that it has has zero mean and unit variance -- Returns ndarray '''

    scaled = preprocessing.scale(a)

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
            objs.append(cPickle.load(f))
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

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, labels, test_size=percentage, random_state=42)

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

def loadData(ecLabels):
    ''' Function for loading the pickled graph fingerprints '''

    # Load Small World Dataset
    SWfeatures, SWlabels = unPickleFingerPrints("Data/SW.pkl")
    SWscaledFeatures = scaleFP(SWfeatures)

    # Load Barabasi Dataset
    BAfeatures, BAlabels = unPickleFingerPrints("Data/BA.pkl")
    BAscaledFeatures = scaleFP(BAfeatures)

    # Load Random Dataset
    ERfeatures, ERlabels = unPickleFingerPrints("Data/ER.pkl")
    ERscaledFeatures = scaleFP(ERfeatures)

    # Load R-MAT Dataset
    RMfeatures, RMlabels = unPickleFingerPrints("Data/RM.pkl")
    RMscaledFeatures = scaleFP(RMfeatures)

    # Load FF Dataset
    FFfeatures, FFlabels = unPickleFingerPrints("Data/FF.pkl")
    FFscaledFeatures = scaleFP(FFfeatures)

    # Concatenate the datasets togther
    features = np.concatenate((SWscaledFeatures, BAscaledFeatures, ERscaledFeatures, RMscaledFeatures, FFscaledFeatures), axis=0)
    labels = np.concatenate((SWlabels, BAlabels, ERlabels, RMlabels, FFlabels), axis=0)
    unScaledFeatures = np.concatenate((SWfeatures, BAfeatures, ERfeatures, RMfeatures, FFfeatures), axis=0)

    # Encode the labels using one hot encoding if required
    if ecLabels:
        labels = encodeLabels(labels)

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

def saveModel(model):
    ''' Save model and weights to disk '''

    print("Saving Model To Disk")

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("DTC-Model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

    model.save_weights("DTC-Weights.h5")

def loadModel():
    ''' Load model from disk '''

    yaml_file = open('DTC-Model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("DTC-Weights.h5")
    print("Loaded model from disk")

    return loaded_model

def plotErrorMatrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    ''' Plot the error matrix for the neural network models '''

    from sklearn.metrics import confusion_matrix

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

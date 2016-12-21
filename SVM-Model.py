# Stephen Bonner 2016 - Durham University
# SVM Classification for graph fingerprint classification
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Utils as ut
from sklearn import svm
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing

def crossValidate(data, labels, model, scoring):
    ''' Pass the data and model to the cross validation function '''
    print "Cross Validation"

    cv = cross_validation.KFold(n=data.shape[0], n_folds=10, shuffle=True)
    scores = cross_validation.cross_val_score(model, data, labels, cv=cv, scoring=scoring, n_jobs=-1)

    return scores

def fitSVM(data, labels, kernel='rbf'):
    ''' Support Vector Classification. The implementations is a based on libsvm.
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    defualt kernel is rbf using one v one approach'''

    model = svm.SVC(C=10, decision_function_shape='ovr', kernel=kernel)
    model = model.fit(data, labels)

    return model

def genSVM(c, kernel):
    ''' Generate an SVM with a specified kernal and C value '''

    clf = svm.SVC(C=c, decision_function_shape='ovr', kernel=kernel)

    return clf

def genLinearSVM(Chere=10e5):
    ''' Generate an SVM with a specified kernal and C value '''

    clf = svm.LinearSVC(dual=False, C=Chere, penalty='l2', multi_class='ovr')

    return clf

def fitLinearSVM(data, labels, Chere=10e5):
    ''' Linear Support Vector Classification.
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    Dual select the algorithm to either solve the dual or primal optimization problem.
    Prefer dual=False when n_samples > n_features. '''

    model = svm.LinearSVC(dual=False, C=Chere, penalty='l2', multi_class='ovr')
    fit = model.fit(data, labels)

    return model, fit

def singleTest(features, labels):
    ''' Run a single test using test train split
    Split the data into testing and training and run initial test '''

    F_train, F_test, l_train, l_test = ut.splitTestTrain(features, labels, 0.4)
    model = fitSVM(F_train, l_train)
    res = model.score(F_test, l_test)
    print res


def createErrorMatrix():
    ''' Create an error matrix for the mulitclass model '''

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import matplotlib
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    import itertools

    oneHotEncode = False
    # Load and Process data
    features, labels, unScaledFeatures = ut.loadData(oneHotEncode)

    # Transform the labels into intergers if not one hot encoding
    if oneHotEncode is False:
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)

    # Split the unencoded labels
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

    clf = genSVM(1, 'linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    res = clf.score(X_test, y_test)
    print res

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print cnf_matrix

    # Load the class labels into the model
    class_names = le.classes_

    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix = cnf_matrix.round(4)
    matplotlib.rcParams.update({'font.size': 22})
    # Plot normalized confusion matrix
    plt.figure()

    ut.plotErrorMatrix(cnf_matrix, classes=class_names, normalize=False)
    plt.show()

def runMulirlabelSVM():
    ''' Run the multimodel SVM testing a variety models '''

    oneHotEncode = False
    # Load and Process data
    features, labels, unScaledFeatures = ut.loadData(oneHotEncode)

    # Extract the global and local scaled features
    localUnScaled = ut.takeJustLocal(unScaledFeatures)
    globalUnScaled = ut.takeJustGlobal(unScaledFeatures)
    localScaled = ut.takeJustLocal(features)
    globalScaled = ut.takeJustGlobal(features)

    # Transform the labels into intergers if not one hot encoding
    if oneHotEncode is False:
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)

    # Cross validate on all Features
    kernels = ['rbf', 'poly', 'linear']
    datasets = [features, unScaledFeatures, localUnScaled, globalUnScaled, localScaled, globalScaled]
    datasetsPrints = ['features', 'unScaledFeatures', 'localUnScaled', 'globalUnScaled', 'localScaled', 'globalScaled']
    count = 0

    for j in datasets:
        for i in kernels:
            print "------------------------------------------------------------------"
            print "Using Kernal " + i + " On Dataset " + datasetsPrints[count]
            clf = genSVM(1, i)

            scores = crossValidate(j, labels, clf, 'f1_micro')
            #print scores
            print("F1-Micro: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

            scores = crossValidate(j, labels, clf, 'accuracy')
            #print scores
            print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

            scores = crossValidate(j, labels, clf, 'recall_micro')
            #print scores
            print("Recall-Micro: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

            scores = crossValidate(j, labels, clf, 'precision_micro')
            #print scores
            print("Precision-Micro: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
        count += 1

def runBinarySVM():
    ''' Run the binary SVM testing a variety models '''

    oneHotEncode = False
    # Load and Process data
    features, labels, unScaledFeatures = ut.loadAnomData(oneHotEncode)

    # Extract the global and local scaled features
    localUnScaled = ut.takeJustLocal(unScaledFeatures)
    globalUnScaled = ut.takeJustGlobal(unScaledFeatures)
    localScaled = ut.takeJustLocal(features)
    globalScaled = ut.takeJustGlobal(features)

    # Transform the labels into intergers if not one hot encoding
    if oneHotEncode is False:
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)

    # Cross validate on all Features
    kernels = ['rbf', 'poly', 'linear']
    datasets = [features, unScaledFeatures, localUnScaled, globalUnScaled, localScaled, globalScaled]
    datasetsPrints = ['features', 'unScaledFeatures', 'localUnScaled', 'globalUnScaled', 'localScaled', 'globalScaled']
    count = 0

    for j in datasets:
        for i in kernels:
            print "------------------------------------------------------------------"
            print "Using Kernal " + i + " On Dataset " + datasetsPrints[count]
            clf = genSVM(1, i)

            scores = crossValidate(j, labels, clf, 'f1_micro')
            #print scores
            print("F1-Micro: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

            scores = crossValidate(j, labels, clf, 'accuracy')
            #print scores
            print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

            scores = crossValidate(j, labels, clf, 'recall_micro')
            #print scores
            print("Recall-Micro: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

            scores = crossValidate(j, labels, clf, 'precision_micro')
            #print scores
            print("Precision-Micro: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

        count += 1

if __name__ == '__main__':

    runMulirlabelSVM()
    print("--------------------RUNING BINARY-------------------------------")
    runBinarySVM()
    createErrorMatrix()

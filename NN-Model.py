# Stephen Bonner 2016 - Durham University
# Code for running the DTC approach

from __future__ import absolute_import
from __future__ import division

import numpy as np
import Utils as ut
import MultiInputModels as mi
import BinaryModels as bi
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import model_from_yaml
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
import pylab as plt
#from keras.utils.visualize_util import plot

# Set the random seed for reproducability
seed = 7
np.random.seed(seed)

def crossValFunc(size, modelfunc, features, labels):
    ''' Setup cross validation '''

    kf = KFold(n=size, n_folds=10, shuffle=True, random_state=seed)
    estimator = KerasClassifier(build_fn=modelfunc, epochs=40, batch_size=256, verbose=0)
    results = cross_val_score(estimator, features, labels, cv=kf)

    return results

def manualCrossValFunc(size, modelfunc, features, labels, nfolds):
    ''' Setup manual cross validation '''

    # Create the kFold validation
    kf = KFold(n=size, n_folds=nfolds, shuffle=True, random_state=seed)

    unTrainedModel = modelfunc
    cvscores = []
    for i, (train, test) in enumerate(kf):
        model = unTrainedModel

        # Fit the model
        model.fit(features[train], labels[train], epochs=200, batch_size=256)
        scores = model.evaluate(features[test], labels[test])
        print("%s: %.5f" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    print "%.5f (+/- %.5f)" % (np.mean(cvscores), np.std(cvscores))

    return cvscores

def gridSearch(features, labels):
    ''' Grid search over paramters '''

    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = np.array([100, 150, 200])
    batches = np.array([5, 10, 20])
    param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    model = KerasClassifier(build_fn=createModel1H)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(features, labels)

    print("Best: %.5f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for params, mean_score, scores in grid_result.grid_scores_:
        print("%.5f (%.5f) with: %r" % (scores.mean(), scores.std(), params))

    return grid_result

def computeModelMetrics(model):
    ''' Function to do a manual cross validation and check precision, recal and f1 '''

    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    # Load data and convert into train and test
    features, labels, unScaledFeatures = ut.loadData(False)

    #features, labels, unScaledFeatures = ut.loadAnomData(False)
    model = mi.createMode3HlWithDropout()

    #model = bi.binaryModel3h()
    #features = unScaledFeatures

    # encode the labels into intergers
    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)

    # Encode to one hot encoding for the training
    #e_labels = np_utils.to_categorical(labels)
    e_labels = labels

    # Create the kFold validation
    kf = KFold(n=features.shape[0], n_folds=10, shuffle=True, random_state=seed)

    unTrainedModel = model
    prscores = []
    rescores = []
    f1scores = []

    for i, (train, test) in enumerate(kf):

        model = unTrainedModel

        # Fit the model
        model.fit(features[train], e_labels[train], epochs=100, batch_size=256, verbose=1)
        scores = model.evaluate(features[test], e_labels[test])

        # Predict the labels from the test data
        y_pred = model.predict_classes(features[test])

        prscores.append(precision_score(labels[test], y_pred, average='micro'))
        rescores.append(recall_score(labels[test], y_pred, average='micro'))
        f1scores.append(f1_score(labels[test], y_pred, average='micro'))


    print "Precicion - %.5f (+/- %.5f)" % (np.mean(prscores), np.std(prscores))
    print "Recall - %.5f (+/- %.5f)" % (np.mean(rescores), np.std(rescores))
    print "F1 - %.5f (+/- %.5f)" % (np.mean(f1scores), np.std(f1scores))

def createTrainingGraph(RunMulti):
    ''' Plot the accuracy and loss of a model and save result using tikz '''

    if RunMulti:
        # Load data
        features, labels, unScaledFeatures = ut.loadData(True)

        model = mi.createMode3HlWithDropout()
        history = model.fit(features, labels, validation_split=0.15, epochs=30, batch_size=2024, verbose=1, shuffle=True)
        print(history.history.keys())
        #plot(model, show_shapes = True, to_file='model.png')

    else:
        # Load Binary data
        features, labels, unScaledFeatures = ut.loadAnomData(True)

        model = bi.binaryModel3h()
        history = model.fit(features, labels, validation_split=0.15, epochs=30, batch_size=2024, verbose=1, shuffle=True)
        print(history.history.keys())
        #plot(model, show_shapes = True, to_file='model.png')


    plt.plot(history.history['acc'], c='b')
    plt.plot(history.history['val_acc'], ls='--', c='r')
    plt.ylim(0.4, 1.09)
    plt.ylabel('Accuracy', labelpad=1)
    plt.xlabel('Epoch', labelpad=1)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    leg = plt.legend(['Train Data', 'Val Data'], loc='lower right')
    for foo in leg.legendHandles:
        foo.set_linewidth(2.0)

    from matplotlib2tikz import save as tikz_save
    tikz_save('accuracy.tex', figureheight='4cm', figurewidth='6cm')
    #plt.show()

    plt.clf()

    plt.plot(history.history['loss'], c='b')
    plt.plot(history.history['val_loss'], ls='--', c='r')
    plt.ylabel('Loss Function Score', labelpad=1)
    plt.xlabel('Epoch', labelpad=1)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    foot = plt.legend(['Train Data', 'Val Data'], loc='top right')
    for toe in leg.legendHandles:
        toe.set_linewidth(2.0)

    #plt.show()
    tikz_save('loss.tex', figureheight='4cm', figurewidth='6cm')

def createErrorMatrix():
    ''' Plot the error matrix for the models '''

    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import matplotlib
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    import itertools

    features, labels, unScaledFeatures = ut.loadData(False)
    model = mi.createMode3HlWithDropout()

    #model = bi.binaryModel1h()
    #features, labels, unScaledFeatures = ut.loadAnomData(ecLabels=True)

    # encode the labels into intergers
    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)

    # Split the unencoded labels
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)

    # Encode to one hot encoding for the training
    Y_Test = np_utils.to_categorical(y_test)
    Y_Train = np_utils.to_categorical(y_train)

    # Train the model
    model.fit(X_train, Y_Train, epochs=2, batch_size=1024, verbose=1)

    # Predict the labels from the test data
    y_pred = model.predict_classes(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print cnf_matrix

    # Load the class labels into the model
    class_names = encoder.classes_

    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix = cnf_matrix.round(4)
    matplotlib.rcParams.update({'font.size': 22})

    # Plot normalized confusion matrix
    plt.figure()
    ut.plotErrorMatrix(cnf_matrix, classes=class_names, normalize=False)
    plt.show()

def runMultiLabel():
    ''' Run the multi class model '''

    # Load data
    features, labels, unScaledFeatures = ut.loadData(True)

    # Extract the global and local scaled features
    localUnScaled = ut.takeJustLocal(unScaledFeatures)
    globalUnScaled = ut.takeJustGlobal(unScaledFeatures)
    localScaled = ut.takeJustLocal(features)
    globalScaled = ut.takeJustGlobal(features)

    # Cross validate on all Features
    nnModels = [mi.createModel1H, mi.createModel2H, mi.createMode3HlWithDropout]
    nnModlesLan = ['1H', '2H', '3H']
    datasets = [features, unScaledFeatures, localUnScaled, globalUnScaled, localScaled, globalScaled]
    datasetsPrints = ['features', 'unScaledFeatures', 'localUnScaled', 'globalUnScaled', 'localScaled', 'globalScaled']
    count = 0

    for i in nnModels:
        dataCount = 0
        for j in datasets:
            print "------------------------------------------------------------------"
            print "Neural Network " + nnModlesLan[count] + " On Dataset " + datasetsPrints[dataCount]
            scores = crossValFunc(j.shape[0], i, j, labels)
            print scores
            print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
            dataCount += 1
        count += 1

def runBinary():
    ''' Run the binary model '''

    # Load data
    features, labels, unScaledFeatures = ut.loadAnomData(True)

    # Extract the global and local scaled features
    localUnScaled = ut.takeJustLocal(unScaledFeatures)
    globalUnScaled = ut.takeJustGlobal(unScaledFeatures)
    localScaled = ut.takeJustLocal(features)
    globalScaled = ut.takeJustGlobal(features)

    # Cross validate on all Features
    nnModels = [bi.binaryModel1h, bi.binaryModel2h, bi.binaryModel3h]
    nnModlesLan = ['1H', '2H', '3H']
    datasets = [features, unScaledFeatures] #, localUnScaled, globalUnScaled, localScaled, globalScaled]
    datasetsPrints = ['features', 'unScaledFeatures', 'localUnScaled', 'globalUnScaled', 'localScaled', 'globalScaled']
    count = 0

    for i in nnModels:
        dataCount = 0
        for j in datasets:
            print "------------------------------------------------------------------"
            print "Neural Network " + nnModlesLan[count] + " On Dataset " + datasetsPrints[dataCount]
            scores = crossValFunc(j.shape[0], i, j, labels, 'accuracy')
            print scores
            print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
            dataCount += 1
        count += 1

if __name__ == '__main__':

    #computeModelMetrics(mi.createMode3HlWithDropout())
    runMultiLabel()
    #runBinary()
    #createTrainingGraph(True)
    #createErrorMatrix()

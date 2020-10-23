import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict, train_test_split
from skorch import NeuralNetClassifier
import numpy as np

from model import DTCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the classifier object
net = NeuralNetClassifier(
    DTCNet,
    max_epochs=100,
    optimizer=torch.optim.RMSprop,
    criterion=torch.nn.CrossEntropyLoss,
    lr=0.0001,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    device=device,
    train_split=None,
)

def computeModelMetrics():
    """ Function to do a manual cross validation and check precision, recall and f1 """

    from sklearn.model_selection import cross_validate

    # Load data
    features, labels, unScaledFeatures = ut.load_data_npy()
    features = features.astype(np.float32)

    mapping = {key:value for key, value in zip(list(set(labels)), range(len(set(labels))))}
    labels = np.array([mapping[x] for x in labels], dtype=np.int64)

    y_pred = cross_validate(net, features, labels, scoring=('recall_micro', 'precision_micro', 'f1_micro', 'accuracy'), cv=10)

    print(y_pred)
    print("Precision - %.5f (+/- %.5f)" % (np.mean(y_pred['test_precision_micro']), np.std(y_pred['test_precision_micro'])))
    print("Recall - %.5f (+/- %.5f)" % (np.mean(y_pred['test_recall_micro']), np.std(y_pred['test_recall_micro'])))
    print("F1 - %.5f (+/- %.5f)" % (np.mean(y_pred['test_f1_micro']), np.std(y_pred['test_f1_micro'])))

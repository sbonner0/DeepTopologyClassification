import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict, train_test_split
from skorch import NeuralNetClassifier

import utils as ut
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
    features, labels, unScaledFeatures = ut.loadData(True)
    features = features.astype(np.float32)

    mapping = {key:value for key, value in zip(list(set(labels)), range(len(set(labels))))}
    labels = np.array([mapping[x] for x in labels], dtype=np.int64)

    y_pred = cross_validate(net, features, labels, scoring=('recall_micro', 'precision_micro', 'f1_micro', 'accuracy'), cv=10)

    print(y_pred)
    print(f"Precision = {np.mean(y_pred['test_precision_micro'])} (+/- {np.std(y_pred['test_precision_micro'])})")
    print(f"Recall = {np.mean(y_pred['test_recall_micro'])} (+/- {np.std(y_pred['test_recall_micro'])})")
    print(f"F1 = {np.mean(y_pred['test_f1_micro'])} (+/- {np.std(y_pred['test_f1_micro'])})")

if __name__=='__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0000001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')   
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()

    computeModelMetrics()
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from torch import nn

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

def run_testset(testloader, model, from_idx, to_idx, device, remove_global=False):

    model.eval()
    test_cumulative_accuracy = 0
    label_list=[]
    prediction_list=[]

    with torch.no_grad():

        for i, data in enumerate(testloader, 0):

            # format the data from the dataloader
            test_inputs, test_labels = data
            test_inputs, test_labels = (test_inputs.to(device), test_labels.to(device))
            test_inputs = test_inputs.float()    

            # remove some features to check the importance
            modified_test_inputs = test_inputs
            modified_test_inputs[:, from_idx:to_idx] = 0.

            if remove_global:
                modified_test_inputs[:, -7:1000] = 0.

            test_outputs = model(modified_test_inputs)
            _, test_predicted = torch.max(test_outputs, 1)    
            test_acc = ut.get_accuracy(test_labels, test_predicted)
            test_cumulative_accuracy += test_acc

    return test_cumulative_accuracy/len(testloader)*100


def runMultiLabel():
    """Run the multi class model using native pytorch"""

    # Load data
    features, labels, unScaledFeatures = ut.loadData(True)
    features = features.astype(np.float32)

    # Extract the global and local scaled features
    localUnScaled = ut.takeJustLocal(unScaledFeatures)
    globalUnScaled = ut.takeJustGlobal(unScaledFeatures)
    localScaled = ut.takeJustLocal(features)
    globalScaled = ut.takeJustGlobal(features)


    mapping = {key:value for key, value in zip(list(set(labels)), range(len(set(labels))))}
    labels = np.array([mapping[x] for x in labels], dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # convert NumPy Array to Torch Tensor
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    # create the data loader for the training set
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=8)

    # create the data loader for the test set
    testset = torch.utils.data.TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=8)

    model = DTCNet()
    model = model.to(device)
    model.train()

    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=0.000001)

    # loop through the required number of epochs
    for epoch in range(50):
        print("Epoch:", epoch)

        # loop through the batches yo!!!
        cumulative_accuracy = 0
        for i, data in enumerate(trainloader, 0):
            # format the data from the dataloader
            inputs, labels = data
            inputs, labels = (inputs.to(device), labels.to(device))
            inputs = inputs.float()
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nll(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the accuracy over the training batch
            _, predicted = torch.max(outputs, 1)
            
            cumulative_accuracy += ut.get_accuracy(labels, predicted)
        print(f"Training Loss: {loss.item()}")
        print(f"Training Accuracy: {(cumulative_accuracy/len(trainloader)*100)}")

    ###### Test the model ######
    model.eval()
    test_cumulative_accuracy = 0
    label_list=[]
    prediction_list=[]

    with torch.no_grad():

        for i, data in enumerate(testloader, 0):

            # format the data from the dataloader
            test_inputs, test_labels = data
            test_inputs, test_labels = (test_inputs.to(device), test_labels.to(device))
            test_inputs = test_inputs.float()    

            # remove some features to check the importance
            modified_test_inputs = test_inputs
            modified_test_inputs[:, 0:25] = 0.

            test_outputs = model(modified_test_inputs)
            _, test_predicted = torch.max(test_outputs, 1)    
            test_acc = ut.get_accuracy(test_labels, test_predicted)
            test_cumulative_accuracy += test_acc

    print("Test Accuracy: %2.5f" % ((test_cumulative_accuracy/len(testloader)*100)))

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

if __name__=='__main__':

    computeModelMetrics()
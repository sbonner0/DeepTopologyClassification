import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import cross_val_predict, train_test_split
from torch import nn

import utils as ut
from model import DTCNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def runMultiLabel(args):
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # create the data loader for the test set
    testset = torch.utils.data.TensorDataset(X_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    model = DTCNet(dropout_level=args.dropout)
    model = model.to(device)
    model.train()

    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # loop through the required number of epochs
    for epoch in range(args.epochs):
        print("Epoch:", epoch)

        # Loop through the batches
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

    with torch.no_grad():

        for i, data in enumerate(testloader, 0):

            # format the data from the dataloader
            test_inputs, test_labels = data
            test_inputs, test_labels = (test_inputs.to(device), test_labels.to(device))
            test_inputs = test_inputs.float()    

            test_outputs = model(test_inputs)
            _, test_predicted = torch.max(test_outputs, 1)    
            test_acc = ut.get_accuracy(test_labels, test_predicted)
            test_cumulative_accuracy += test_acc

    print("Test Accuracy: %2.5f" % ((test_cumulative_accuracy/len(testloader)*100)))

if __name__=='__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0000001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')   
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()

    runMultiLabel(args)
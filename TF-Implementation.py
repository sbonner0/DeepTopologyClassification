# Stephen Bonner 2016 - Durham University
# This is a Tensorflow implementation of the DTC approach with both native and TFslim versions

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import Utils as ut

# Load the data and perform the test train split
features, labels, unScaledFeatures = ut.loadData(True)
X_train, X_test, y_train, y_test = ut.splitTestTrain(features, labels, 0.3)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 64 # 2nd layer number of features
n_input = 54 # GFP Input Dimensions
n_classes = 5

def dataIterator():
    ''' A simple data iterator '''

    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(X_train))
        np.random.shuffle(idxs)
        shuf_features = X_train[idxs]
        shuf_labels = y_train[idxs]
        for batch_idx in range(0, len(X_train), batch_size):
            features_batch = shuf_features[batch_idx:batch_idx+batch_size]
            features_batch = features_batch.astype("float32")
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield features_batch, labels_batch


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input], name="x")
y = tf.placeholder(tf.float32, [None, n_classes], name="y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# Create model
def denseLayers(x):
    ''' Dense layers using manual weight and biase creation '''

    with tf.variable_scope('hidden_layer_1'):
        weights = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
        biases = tf.Variable(tf.random_normal([n_hidden_1]))
        layer_1 = tf.nn.relu(tf.matmul(x, weights) + biases)
        layer_1 = tf.nn.dropout(layer_1, keep_prob)

    with tf.variable_scope('hidden_layer_2'):
        weights = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
        biases = tf.Variable(tf.random_normal([n_hidden_2]))
        layer_2 = tf.nn.relu(tf.matmul(layer_1, weights) + biases)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)

    with tf.variable_scope('hidden_layer_3'):
        weights = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]))
        biases = tf.Variable(tf.random_normal([n_hidden_3]))
        layer_3 = tf.nn.relu(tf.matmul(layer_2, weights) + biases)
        layer_3 = tf.nn.dropout(layer_3, keep_prob)

    with tf.variable_scope('softmax_linear'):
        weights = tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
        biases = tf.Variable(tf.random_normal([n_classes]))
        logits = tf.matmul(layer_3, weights) + biases

    return logits

# Creating the model in TF Slim for comparison. Allows easy use of Glorot initilaisation.
# We also use L2 regularization to help with over fitting
def denseLayersTFSlim(x):
    ''' Dense layers using TFSlim package'''

    layer_1 = slim.fully_connected(x,
        n_hidden_1,
        activation_fn=slim.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=slim.initializers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(0.000001),
        biases_initializer=slim.init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)

    layer_2 = slim.fully_connected(layer_1,
        n_hidden_2,
        activation_fn=slim.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=slim.initializers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(0.000001),
        biases_initializer=slim.init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)

    layer_3 = slim.fully_connected(layer_2,
        n_hidden_3,
        activation_fn=slim.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=slim.initializers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(0.000001),
        biases_initializer=slim.init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    logits = slim.fully_connected(layer_3,
        n_classes,
        activation_fn=None,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=slim.initializers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(0.000001),
        biases_initializer=slim.init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None)

    return logits

# Loss function
def lossFunction(logits):
    ''' Loss function using Caterorigal Cross entropy '''

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name='crossEnt')
    loss = tf.reduce_mean(cross_entropy, name='crossEnt_mean')

    return loss

#logits = denseLayers(x)
logits = denseLayersTFSlim(x)

# Define loss and add the L2 regularazation and optimizer
loss = lossFunction(logits)
loss = loss + tf.losses.get_total_loss()

tf.summary.scalar('loss', loss)

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
global_step = tf.Variable(0, name='globalStep', trainable=False)
grads = optimizer.compute_gradients(loss)
apply_gradients = optimizer.apply_gradients(grads, global_step)
#for grad,var in grads:
 #  tf.summary.histogram('/gradient', grad)

# initalise all the variables
init_op = tf.global_variables_initializer()
iter_ = dataIterator()

# Launch the tf graph
with tf.Session() as sess:
    sess.run(init_op)

    # Setup tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/tensorboard' + '/train',
                                      sess.graph)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.

        # Calculate the total batch size based on the size of the input data
        total_batch = int(X_train.shape[0]/batch_size)

        # Loop over all batches
        for i in range(total_batch):

            batch_x, batch_y = iter_.next()

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary = sess.run([apply_gradients, loss, merged], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: 0.8})

            train_writer.add_summary(summary, epoch*total_batch+i)

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

    print("Training Complete!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy:", accuracy.eval({x: X_test, y: y_test, keep_prob: 1}))

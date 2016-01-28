# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import cPickle as pickle
import numpy as np
import tensorflow as tf
import time

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print 'Training set', train_dataset.shape, train_labels.shape
    print 'Validation set', valid_dataset.shape, valid_labels.shape
    print 'Test set', test_dataset.shape, test_labels.shape

image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print 'Training set', train_dataset.shape, train_labels.shape
print 'Validation set', valid_dataset.shape, valid_labels.shape
print 'Test set', test_dataset.shape, test_labels.shape

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 128
hidden_units_1 = 1024
hidden_units_2 = 300
hidden_units_3 = 50

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = {
    'h1': tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units_1], stddev=.1)),
    'h2': tf.Variable(tf.truncated_normal([hidden_units_1, hidden_units_2], stddev=.1)),
    'h3': tf.Variable(tf.truncated_normal([hidden_units_2, hidden_units_3], stddev=.1)),
    'out': tf.Variable(tf.truncated_normal([hidden_units_3, num_labels], stddev=.25))
    }
    biases = {
    'b1': tf.Variable(tf.zeros([hidden_units_1])),
    'b2': tf.Variable(tf.zeros([hidden_units_2])),
    'b3': tf.Variable(tf.zeros([hidden_units_3])),
    'out': tf.Variable(tf.zeros([num_labels]))
        }
    beta = tf.constant(.0000001)
    keep_prob = tf.constant(1.0)

    # Training computation.
    layer_1 = tf.nn.tanh(tf.matmul(tf_train_dataset, weights['h1']) + biases['b1'])
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['h2']) + biases['b2'])
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['h3']) + biases['b3'])
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    logits =  tf.matmul(layer_3, weights['out']) + biases['out']
    loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
    beta * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
            tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out']))

    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, step, len(train_dataset), .95)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    val_layer_1 = tf.nn.tanh(tf.matmul(tf_valid_dataset, weights['h1']) + biases['b1'])
    val_layer_2 = tf.nn.tanh(tf.matmul(val_layer_1, weights['h2']) + biases['b2'])
    val_layer_3 = tf.nn.tanh(tf.matmul(val_layer_2, weights['h3']) + biases['b3'])
    valid_prediction = tf.nn.softmax(
    tf.matmul(val_layer_3, weights['out']))# + biases['out'])
    test_layer_1 = tf.nn.tanh(tf.matmul(tf_test_dataset, weights['h1']) + biases['b1'])
    test_layer_2 = tf.nn.tanh(tf.matmul(test_layer_1, weights['h2']) + biases['b2'])
    test_layer_3 = tf.nn.tanh(tf.matmul(test_layer_2, weights['h3']) + biases['b3'])
    test_prediction = tf.nn.softmax(
    tf.matmul(test_layer_3, weights['out']) + biases['out'])


num_steps = 3001
t0 = time.time()

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print "Initialized"
    for step in xrange(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print "Minibatch loss at step", step, ":", l
            print "Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels)
            print "Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels)
    print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels)
print time.time() - t0

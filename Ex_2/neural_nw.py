#source: https://pythonprogramming.net/tensorflow-deep-neural-network-machine-learning-tutorial/
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import load_ORL_faces as ORL
import matplotlib.pyplot as plt
import numpy as np


mnist = input_data.read_data_sets("data/", one_hot = True)

num_iteration_mnist = 10000
num_iteration_orl = 2000
n_nodes_hl1 = 500
n_nodes_hl2 = 500
batch_size = 100

x = tf.placeholder('float', [None, None])
#x = tf.placeholder('float', [None, 10304])
y = tf.placeholder('int64')


def neural_network_model(data, data_size, n_classes):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([data_size, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']

    return output

def train_mnist_network(x):
    n_classes = 10
    prediction = neural_network_model(x, 784, n_classes)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        d = {}
        for i in range(num_iteration_mnist):
            epoch_loss = 0
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

            if i%100 == 0:
                print 'Iteration ', i, 'completed out of',num_iteration_mnist,'loss:',epoch_loss
                d[i] = epoch_loss

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
        
        lists = sorted(d.items()) # sorted by key, return a list of tuples
        x1, y1 = zip(*lists) # unpack a list of pairs into two tuples
        plt.plot(x1,y1)
        plt.show()
        
def part_a():
    
    train_mnist_network(x)
    
def visualize(trainX):
    trainX = np.array(trainX, dtype='uint8')
    #trainY = np.array(trainY, dtype='uint8')
    #print trainX.shape[0], trainY.shape[0]
    fig = plt.figure()
    for i in range(4):
        a = fig.add_subplot(2, 2, i+1)
        pixels = trainX[i].reshape((112, 92))
        plt.imshow(pixels, cmap='gray')
        
    plt.show()

def train_ORL_network(x, trainX, trainY, testX, testY):
    n_classes = 20
    prediction = neural_network_model(x, 10304, n_classes)
    
    cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        d = {}
        for i in range(num_iteration_orl):
            epoch_loss = 0
            epoch_x, epoch_y = ORL.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

            if i%20 == 0:
                print 'Iteration ', i, 'completed out of',num_iteration_orl,'loss:',epoch_loss
                d[i] = epoch_loss

        correct = tf.equal(tf.argmax(prediction, 1), y)

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print 'Accuracy:',accuracy.eval({x:testX, y:testY})
        
        lists = sorted(d.items()) # sorted by key, return a list of tuples
        x1, y1 = zip(*lists) # unpack a list of pairs into two tuples
        plt.plot(x1,y1)
        plt.show()
    
def part_b():
    trainX, trainY, testX, testY = ORL.read_data()   
    visualize(trainX)       
    train_ORL_network(x, trainX, trainY, testX, testY)
        

if __name__ == '__main__':
    part_a()
    part_b()
    
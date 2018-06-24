import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import load_ORL_faces as ORL

n_classes = 20
n_input = 10304

# Add convolution layer
def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([5, 5, size_in, size_out], stddev=0.01), name="W")
        b = tf.Variable(tf.random_normal([size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Add fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([size_in, size_out], stddev=0.01), name="W")
        b = tf.Variable(tf.random_normal([size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        return act

def out_layer(input, size_in, size_out, name="out"):
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([size_in, size_out], stddev=0.01), name="W")
        b = tf.Variable(tf.random_normal([size_out]), name="B")
        act = tf.matmul(input, w) + b
        return act

def cnn_model(x,y):
    
   # Reshape input to a 4D tensor 
    x = tf.reshape(x, shape=[-1, 112, 92, 1])
    
    conv1 = conv_layer(x, 1, 32, "conv1")
    conv2 = conv_layer(conv1, 32, 64, "conv2") 
    
    flatten = tf.reshape(conv2, [-1, 28*23*64])
    
    fc1 = fc_layer(flatten, 28*23*64, 1024, "fc1")
        
    logits = out_layer(fc1, 1024, n_classes, "out")
    
    pred = tf.argmax(logits, axis=1)
    
    return pred, logits

def cnn_model_dropout(x,y):
    
   # Reshape input to a 4D tensor 
    x = tf.reshape(x, shape=[-1, 112, 92, 1])
    
    conv1 = conv_layer(x, 1, 32, "conv1")
    conv2 = conv_layer(conv1, 32, 64, "conv2") 
    
    flatten = tf.reshape(conv2, [-1, 28*23*64])
    
    fc1 = fc_layer(flatten, 28*23*64, 1024, "fc1")
    
    #drop out layer
    fc_drop = tf.nn.dropout(fc1, keep_prob=0.5)
    
    logits = out_layer(fc_drop, 1024, n_classes, "out")
    
    pred = tf.argmax(logits, axis=1)
    
    return pred, logits

def compute_loss(logits,y):
    # Define loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y) )
    
    tf.summary.scalar('loss',loss)
    return loss

def cal_error(pred,y):
    with tf.name_scope('error'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(pred, y)
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))# Compute accuracy
        with tf.name_scope('error'):
          error = 1- accuracy
   
    tf.summary.scalar('error', error)
    return error, accuracy

def train_network(trainX, trainY, testX, testY, lr, drop):
    
    x = tf.placeholder("float", [None, n_input], name='x')
    y = tf.placeholder("int64",  name='y')
      
    if drop:
        pred, logits = cnn_model_dropout(x, y)
    else:
        pred, logits = cnn_model(x, y)
        
    loss = compute_loss(logits,y)
    error, accuracy = cal_error(pred,y)
    

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        
    #'''
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
        
    merged = tf.summary.merge_all()
    
    if drop:
        train_writer = tf.summary.FileWriter( './train_drop_' + str(lr), sess.graph)
        test_writer  = tf.summary.FileWriter('./test_drop_' + str(lr))
    else:
        train_writer = tf.summary.FileWriter( './train_' + str(lr), sess.graph)
        test_writer  = tf.summary.FileWriter('./test_' + str(lr))
    tf.reset_default_graph()
    #'''    
    #'''   
    print 'For lr ' + str(lr) + ':'
    for i in range(2000):
        epoch_x, epoch_y = ORL.next_batch(100) # Fetch batch
    
        if (i % 50) == 0:  # Record summaries ( loss etc) and test-set accuracy          
            summary, err = sess.run([merged, error],feed_dict={x: testX, y: testY})# Run session to compute summary and eror
            test_writer.add_summary(summary, i)
            print('Classification Error after Itration %s : %s' % (i,err))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict={x: epoch_x, y: epoch_y})
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
        
    print ("Accuracy using tensor flow is:")
    print(sess.run(accuracy, feed_dict={x: testX, y: testY}))
        
    writer = tf.summary.FileWriter('cnn_model',sess.graph)
    writer.add_graph(sess.graph)
    #'''
        
if __name__ == '__main__':
    
    trainX, trainY, testX, testY = ORL.read_data()
    
    #(a)With diff learning rate and without dropout
    learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001]    
    for lr in learning_rate:
        train_network(trainX, trainY, testX, testY, lr, False)
        
    print ' Training with dropout'
    train_network(trainX, trainY, testX, testY, 0.001, True)
    
    
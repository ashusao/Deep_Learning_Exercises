import numpy as np  
 

# load data
data = np.load('ORL_faces.npz')
trainX = data['trainX']
trainY = data['trainY']
testX = data['testX']
testY = data['testY']
 
epochs_completed = 0
index_in_epoch = 0
num_examples = trainX.shape[0]

# source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
def next_batch(batch_size, shuffle=True):
    
    global index_in_epoch
    global epochs_completed
    global num_examples
    global trainX
    global trainY
    global testX
    global testY
    
    start = index_in_epoch
    # Shuffle for the first epoch
    if epochs_completed == 0 and start == 0:
        perm0 = np.arange(num_examples)
        np.random.shuffle(perm0)
        trainX = trainX[perm0]
        trainY = trainY[perm0]
      
    # Go to the next epoch
    if start + batch_size > num_examples:
      # Finished epoch
        epochs_completed += 1
        # Get the rest examples in this epoch
        rest_num_examples = num_examples - start
        images_rest_part = trainX[start:num_examples]
        labels_rest_part = trainY[start:num_examples]
        # Shuffle the data
        if shuffle:
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            trainX = trainX[perm]
            trainY = trainY[perm]
          # Start next epoch
        start = 0
        index_in_epoch = batch_size - rest_num_examples
        end = index_in_epoch
        images_new_part = trainX[start:end]
        labels_new_part = trainY[start:end]
        return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
        index_in_epoch += batch_size
        end = index_in_epoch
        return trainX[start:end], trainY[start:end]
    
def read_data():
   return trainX, trainY, testX, testY
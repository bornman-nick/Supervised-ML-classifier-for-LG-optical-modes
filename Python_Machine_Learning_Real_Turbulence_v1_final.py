# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Nicholas Bornman
"""



## SAVE THE MODEL AT CERTAIN STEPS
## DO WE TRAIN THE MODEL AFTER USING THE VALIDATION BATCH?




# These are the packages you will need.
import numpy as np
import tensorflow as tf
import cv2
import os
import random as rand
import sys
import time as time
from tkinter import filedialog
import tkinter




# Define the the number of classes we have for our input images
number_classes = 6

# Number of colour channels in images
num_channels = 1

# Choose which type of model to use: simple model (set to True) or convolutional neural network (set to False)
simple_model = False

# Set whether the TFRecords files exist yet or not
tfrecordexist = True

# Percentage of data we wish to use in our training, validation and testing
percentage = 0.1

# Batch size we would like to use when training the model
batches_size = 100

# Size we would like the load_image programme to change our image size to, from the 
# input number of pixels to p by p pixels
# p must be a multiple of four, unless you change the later code to include a different
# number of pool operations in the convolutional neural network model
p = 40

#Define the percentage of the data you want to partition into the training set,
# the validation set, and the training set. These percentages must obviously
# sum to 1.
training_percentage = 0.6
validation_percentage = 0.2
test_percentage = 1 - (training_percentage + validation_percentage)



# File directory paths for input image data. Change these directories to the directories where your input image data
# is, and add more directories if you have more folders.


mode10_data_path = "C:\\Users\\Nick\\Desktop\\University of Waterloo\\Machine Learning Project\\Raw data\\Turbulence Data\\Frames\l=1,p=0\\"
mode11_data_path = "C:\\Users\\Nick\\Desktop\\University of Waterloo\\Machine Learning Project\\Raw data\\Turbulence Data\\Frames\l=1,p=1\\"
mode20_data_path = "C:\\Users\\Nick\\Desktop\\University of Waterloo\\Machine Learning Project\\Raw data\\Turbulence Data\\Frames\l=2,p=0\\"
mode21_data_path = "C:\\Users\\Nick\\Desktop\\University of Waterloo\\Machine Learning Project\\Raw data\\Turbulence Data\\Frames\l=2,p=1\\"
mode30_data_path = "C:\\Users\\Nick\\Desktop\\University of Waterloo\\Machine Learning Project\\Raw data\\Turbulence Data\\Frames\l=3,p=0\\"
mode31_data_path = "C:\\Users\\Nick\\Desktop\\University of Waterloo\\Machine Learning Project\\Raw data\\Turbulence Data\\Frames\l=3,p=1\\"



# The number of images in each path
length_mode10 = len(os.listdir(mode10_data_path))
length_mode11 = len(os.listdir(mode11_data_path))
length_mode20 = len(os.listdir(mode20_data_path))
length_mode21 = len(os.listdir(mode21_data_path))
length_mode30 = len(os.listdir(mode30_data_path))
length_mode31 = len(os.listdir(mode31_data_path))


# Create lists of strings representing the individual image paths in each input directory, as well as
# separate lists of their labels
addrs_mode10 = [os.path.join(mode10_data_path,item) for item in os.listdir(mode10_data_path)]
labels_mode10 = [0]*length_mode10
addrs_mode11 = [os.path.join(mode11_data_path,item) for item in os.listdir(mode11_data_path)]
labels_mode11 = [1]*length_mode11
addrs_mode20 = [os.path.join(mode20_data_path,item) for item in os.listdir(mode20_data_path)]
labels_mode20 = [2]*length_mode20
addrs_mode21 = [os.path.join(mode21_data_path,item) for item in os.listdir(mode21_data_path)]
labels_mode21 = [3]*length_mode21
addrs_mode30 = [os.path.join(mode30_data_path,item) for item in os.listdir(mode30_data_path)]
labels_mode30 = [4]*length_mode30
addrs_mode31 = [os.path.join(mode31_data_path,item) for item in os.listdir(mode31_data_path)]
labels_mode31 = [5]*length_mode31



# The above lists are paired up (an image path with its label), combined, and shuffled
mode10 = list(zip(addrs_mode10,labels_mode10))
mode11 = list(zip(addrs_mode11,labels_mode11))
mode20 = list(zip(addrs_mode20,labels_mode20))
mode21 = list(zip(addrs_mode21,labels_mode21))
mode30 = list(zip(addrs_mode30,labels_mode30))
mode31 = list(zip(addrs_mode31,labels_mode31))

total_set = list(set().union(mode10,mode11,mode20,mode21,mode30,mode31))
rand.shuffle(total_set)
addresses, labels = zip(*total_set)



train_addrs = addresses[0:int(training_percentage*len(addresses))]
train_labels = labels[0:int(training_percentage*len(labels))]
val_addrs = addresses[int(training_percentage*len(addresses)):int((training_percentage + validation_percentage)*len(addresses))]
val_labels = labels[int(training_percentage*len(labels)):int((training_percentage + validation_percentage)*len(labels))]
test_addrs = addresses[int((training_percentage + validation_percentage)*len(addresses)):]
test_labels = labels[int((training_percentage + validation_percentage)*len(labels)):]



if num_channels == 1:
    chan = 0
elif num_channels == 3:
    chan = 1

def load_image(addr):
    img = cv2.imread(addr,chan)
    img = cv2.resize(img, (p, p), interpolation = cv2.INTER_AREA)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


root = tkinter.Tk()
root_directory = filedialog.askdirectory(initialdir = "/", title = "Select directory to access TFRecords files")
root.withdraw()
print("Your selected directory is: " + root_directory)


training_filename = root_directory + "/TFRecords_training_file"
val_filename = root_directory + "/TFRecords_val_file"
test_filename = root_directory + "/TFRecords_test_file"


if not tfrecordexist:
    
    writer_train = tf.python_io.TFRecordWriter(training_filename)
    writer_val = tf.python_io.TFRecordWriter(val_filename)
    writer_test = tf.python_io.TFRecordWriter(test_filename)
    
    print("Creating training set TFRecords file.")
    
    for i in range(int(len(train_addrs)*percentage)):
        img = load_image(train_addrs[i])
        label = train_labels[i]
        feature = {'label': _int64_feature(label),'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_train.write(example.SerializeToString())
    
    writer_train.close()
    sys.stdout.flush()
    
    print("Training set file created. Total number of training images: %d. Creating validation set TFRecords file." % (int(len(train_addrs)*percentage)))
    
    for i in range(int(len(val_addrs)*percentage)):
        img = load_image(val_addrs[i])
        label = val_labels[i]
        feature = {'label': _int64_feature(label),'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_val.write(example.SerializeToString())
    
    writer_val.close()
    sys.stdout.flush()
    
    print("Validation set file created. Total number of validation images: %d. Creating test set TFRecords file." % (int(len(val_addrs)*percentage)))
    
    for i in range(int(len(test_addrs)*percentage)):
        img = load_image(test_addrs[i])
        label = test_labels[i]
        feature = {'label': _int64_feature(label),'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer_test.write(example.SerializeToString())
    
    writer_test.close()
    sys.stdout.flush()
    
    print("Test set file created. Total number of test images: %d." % (int(len(test_addrs)*percentage)))



# Pixel dimensions of images
height = p
width = p



# Function to tell TensorFlow how to read a single image from input file, including its features
def getImage(filename):
    
    feature = {'image': tf.FixedLenFeature([], tf.string),'label': tf.FixedLenFeature([], tf.int64)}
    
    # Convert filenames to a queue
    filenameQ = tf.train.string_input_producer([filename], num_epochs=None)
 
    # Object to read records
    reader = tf.TFRecordReader()

    # Read the full set of features for a single example. Can also read the dictionary keys if you wish 
    _, example = reader.read(filenameQ)

    # Parse the full example into its' component features.
    features = tf.parse_single_example(example, features = feature)
        
    label = features['label']
    image_buffer = features['image']

    image = tf.decode_raw(image_buffer, tf.float32)

    # Cast image into a single array of length height*width*num_channels instead of a 2D array
    image = tf.reshape(image,[width*height*num_channels])

    # Re-define label as a "one-hot" vector
    label = tf.stack(tf.one_hot(label, number_classes))

    return label, image



# Associate label and image objects with the corresponding features read from a single example in the
# training data file
label, image = getImage(training_filename)

# Similarly for the validation data
vlabel, vimage = getImage(val_filename)

# Similarly for the test data
testlabel, testimage = getImage(test_filename)


# Associate imageBatch and labelBatch objects with a randomly selected batch of labels and images
# respectively. Can change the batch size here is you wish
imageBatch, labelBatch = tf.train.shuffle_batch([image, label], batch_size = batches_size, capacity = 2000, min_after_dequeue = 1000, allow_smaller_final_batch = True)

# Similarly for the validation data 
vimageBatch, vlabelBatch = tf.train.shuffle_batch([vimage, vlabel], batch_size = batches_size, capacity = 2000, min_after_dequeue = 1000, allow_smaller_final_batch = True)

# Similarly for the test data. Note, however, that this gets sequential batches of the
# test data images, and does not randomly shuffle them like for the other two batches. This
# appears to be quicker, and shuffling isn't needed, like in the other two sets.
testimageBatch, testlabelBatch = tf.train.batch([testimage, testlabel], batch_size = batches_size, capacity = 2000, allow_smaller_final_batch = True)



# Class to actually run the Tensorflow operations
sess = tf.InteractiveSession()

# Placeholder variables for the input images, x, and their corresponding input labels, y_
x = tf.placeholder(tf.float32, [None, width*height*num_channels], name = 'x')
y_true = tf.placeholder(tf.float32, [None, number_classes], name = 'y_true')
y_true_class = tf.argmax(y_true,1)

# Now we simply copy, almost verbatim, the two algorithms given on the Tensorflow MNIST tutorial

if simple_model:
    # Run simple model y = Wx+b given in TensorFlow "MNIST" tutorial
    print("Running simple model y = Wx + b")
    
    W = tf.Variable(tf.zeros([width*height*num_channels, number_classes]))
    
    b = tf.Variable(tf.zeros([number_classes]))
    
    y_conv = tf.matmul(x, W) + b
    
    y_pred = tf.nn.softmax(y_conv, name = "y_pred")
    
else:
    # Run convolutional neural network model given in "Expert MNIST" TensorFlow tutorial
    
    def create_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))
    
    def create_biases(size):
        return tf.Variable(tf.constant(0.05,shape = [size]))
    
    
    def conv2d(input, filter):
        return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    def create_convolutional_layer(input,num_input_channels,conv_filter_size,num_filters,relufirst):
        
        ## We shall define the weights that will be trained using create_weights function.
        weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        
        ## We create biases using the create_biases function. These are also trained.
        biases = create_biases(num_filters)
        
        ## Creating the convolutional layer
        layer = conv2d(input = input, filter = weights) + biases
        
        if relufirst == True:
            # First we feed to conv2d output to a relu function
            layer = tf.nn.relu(layer)
            # Then we max pool the result
            layer = max_pool_2x2(input = layer)
        else:
            ## We shall be using max-pooling
            layer = max_pool_2x2(input = layer)
            ## Output of pooling is fed to Relu which is the activation function for us
            layer = tf.nn.relu(layer)
        
        return layer
    
    def create_flatten_layer(layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])
        return layer
    
     
    def create_fc_layer(input,num_inputs,num_outputs,use_relu=True):
        
        #Let's define trainable weights and biases.
        weights = create_weights(shape=[num_inputs, num_outputs])
        biases = create_biases(num_outputs)
        
        layer = tf.matmul(input, weights) + biases
        
        if use_relu:
            layer = tf.nn.relu(layer)
        
        return layer
    
    print("Running convolutional neural network model")
    
    
    x_image = tf.reshape(x, [-1,width,height,num_channels])
    
    
    ## ACUTAL MODEL ARCHITECTURE, between (*---*)
    
    ## (*---*)
    
    
    number_features1 = 16 # Number of features/filters in first conv layer
    number_features2 = 32 # Number of features/filters in second conv layer
    number_neurons = 512 # Number of neurons in fully-connected layer
    
    
    layer_conv1 = create_convolutional_layer(x_image,num_channels,5,number_features1,True)
    
    layer_conv2 = create_convolutional_layer(layer_conv1,number_features1,5,number_features2,True)
    
    # We need to check that our dimensions are a multiple of 4
    if (width % 4 != 0 or height % 4 != 0):
        print("Error: width and height must be a multiple of 4")
        sys.exit(1)
    
    
    layer_flat3 = create_flatten_layer(layer_conv2)
    
    # int((width/4)*(height/4)*number_features2)
    # layer_flat3.get_shape()[1:4].num_elements()
    
    layer_3_size = layer_flat3.get_shape()[1:4].num_elements()
    
    
    layer_full4 = create_fc_layer(layer_flat3,layer_3_size,number_neurons,True)
    
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    layer_full4drop = tf.nn.dropout(layer_full4, keep_prob)
    
    W_full4drop = create_weights([number_neurons,number_classes])
    b_full4drop = create_biases(number_classes)
    
    y_conv = tf.matmul(layer_full4drop,W_full4drop) + b_full4drop
    
    y_pred = tf.nn.softmax(y_conv, name = "y_pred")
    
    ## (*---*)



# The predicted class
y_pred_class = tf.argmax(y_pred,1)



# Cross entropy and cost function. Can change to just one function instead of the two if's if you want
if simple_model:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
else:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_conv, labels = y_true)

cost = tf.reduce_mean(cross_entropy)


# Define the training step which minimises cross entropy. Can change tf.train
# optimisation algorithm
optimiser = tf.train.AdamOptimizer(0.0001).minimize(cost)

# Returns a boolean 'True' or 'False' if the argmax indexes are equal
correct_prediction = tf.equal(y_true_class, y_pred_class)

# Get the mean of all entries in correct prediction; the higher, the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# saver = tf.train.Saver()




# Initialize the variables
sess.run(tf.global_variables_initializer())

# Start the threads used for reading files (I don't really know what
# this is for, but hey)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)










# Start the actual training. You can alter the number of steps if you want

print("Training chosen model")


number_steps = int(len(train_addrs)*percentage/batches_size)

for i in range(number_steps):

    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    
    vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
    
    # Run the training step with the i'th batch of images
    if simple_model:
      optimiser.run(feed_dict={x: batch_xs, y_true: batch_ys})
    else:
      optimiser.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})

    # Perform the validation using a batch of validation set images
    # every so many runs. Then train the model using the validation set
    if i % int((number_steps/10)*(validation_percentage/training_percentage)) == 0: 

      # Get a validation batch.
      if simple_model:
        val_accuracy = sess.run(accuracy, feed_dict = {x: vbatch_xs, y_true: vbatch_ys})
        percen = int(i*100/number_steps)
        print("Percentage complete: %d; Accuracy using validation set batch: %g" % (percen, val_accuracy))
        # optimiser.run(feed_dict = {x: vbatch_xs, y_true: vbatch_ys})
      else:
        val_accuracy = sess.run(accuracy, feed_dict = {x: vbatch_xs, y_true: vbatch_ys, keep_prob: 1.0})
        percen = int(i*100/number_steps)
        print("Percentage complete: %d; Accuracy using validation set batch: %g" % (percen, val_accuracy))
        # optimiser.run(feed_dict = {x: vbatch_xs, y_true: vbatch_ys, keep_prob: 0.5})
    # saver.save(sess, 'modes_of_light_model')





# saver1 = tf.train.import_meta_graph('modes_of_light_model.meta')

# saver1.restore(sess, tf.train.latest_checkpoint('./'))



# Now, test the final trained model using the test set


print("Testing chosen model.")


test_steps = int(len(test_addrs)*percentage/batches_size)
test_accuracy = 0

for i in range(test_steps):
    
    # Get i'th batch of test data
    test_xs, test_ys = sess.run([testimageBatch, testlabelBatch])
    
    if simple_model:
        test_accuracy += sess.run(accuracy, feed_dict = {x: test_xs, y_true: test_ys})
    else:
        test_accuracy += sess.run(accuracy, feed_dict = {x: test_xs, y_true: test_ys, keep_prob: 1.0})
    
    if i % int(round(test_steps/10)) == 0:
        print("Test progress: %g" % (i*100/(test_steps)))

print("Test progress: 100")
        
print("Accuracy using test set: %g" % (test_accuracy/test_steps))

print("Training and testing complete.")

# Define a function, with the input being an image's address, which you want to classify with the
# machine.

def categorise(addr,pic):
    images = load_image(addr)
    
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    img1 = np.reshape(images, [1,width*height*num_channels])
    
    graph = tf.get_default_graph()
    
    # Feed the images into the placeholders
    
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    
    y_test_images = np.zeros((1, number_classes))
    
    if simple_model:
        feed_dict_testing = {x: img1, y_true: y_test_images}
    else:
        feed_dict_testing = {x: img1, y_true: y_test_images, keep_prob: 1}
    
    result = sess.run(y_pred, feed_dict = feed_dict_testing)
    
    a = sess.run(tf.argmax(result[0]))
    
    p = a
    
    if p == 0:
        p = 1,0
    elif p == 1:
        p = 1,1
    elif p == 2:
        p = 2,0
    elif p == 3:
        p = 2,1
    elif p == 4:
        p = 3,0
    elif p == 5:
        p = 3,1
    
    c = round(100*result[0][a],2)
    
    print("Most likely mode: l,p = %s | Probability: %g" % (p,c))
    
    if pic == True:
        img = cv2.imread(addr,0)
        cv2.imshow('Mode: ' + str(a) + " | Probability: " + str(c),img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    



# Run the prediction algorithm for an image from any of the three sets, and
# time how long it takes to categorise the 20 images.

start = time.time()

for i in range(20):
    categorise(test_addrs[i],False)

end = time.time()

print(end - start)


categorise(test_addrs[50],True)


# Finally, close everything
    
coord.request_stop()
coord.join(threads)
sess.close()


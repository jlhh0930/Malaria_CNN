#Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score

#Check that the two image sets exist
#print(os.listdir("../malaria/cell_images/cell_images"))

#Get the image sets and save them into infected and uninfected
infected = os.listdir('../malaria/cell_images/cell_images/Parasitized')
uninfected = os.listdir('../malaria/cell_images/cell_images/Uninfected')

#Variables to hold stuff - data will hold the images, labels will hold 1 or 0 for each image
data = []
labels = []

#function to take the images from the csv,
#manipulate them as needed, and save them for later use
def store_images_from_csv(imageType, filePath, isInfected):
    path = str(filePath)
    images = os.listdir(path)
    inf = isInfected
    for i in images:
        try:
            # Get the image
            image = cv2.imread(path + i)
            # store the image as an array
            image_array = Image.fromarray(image, 'RGB')
            # resize the image to something more manageable
            resize_img = image_array.resize((50, 50))
            # rotate 45 and 75 degrees
            rotate45 = resize_img.rotate(45)
            rotate75 = resize_img.rotate(75)
            #blur infected images only - blurring helps reduce noise
            #in the image so we can (hopefully) get a more accurate
            #classification and prediction
            if inf == 1:
                blur = cv2.blur(np.array(resize_img), (10, 10))
            # append all four image arrays to the data dictionary
            data.append(np.array(resize_img))
            data.append(np.array(rotate45))
            data.append(np.array(rotate75))
            #append the blurred image for infected cells
            if inf == 1:
                data.append(np.array(blur))
            # add labels - 1 denotes infected, 0 marks uninfected
            labels.append(inf)
            labels.append(inf)
            labels.append(inf)
            #add the extra blurred label if it is an infected cell
            if inf == 1:
                labels.append(inf)

        except AttributeError:
            print('Error storing image ' + imageType + ' ' + i)
            return 0

#Basic overview of a cnn (convolutional neural network):
#A neural network takes data from multiple neurons and
#feeds it through multiple processing layers before
#giving a prediciton.
#A convolutional neural network only connects the neurons
#that are close to one another so that the spacial relationships
#are maintained.  This is necessary for an image classifaction
#because images are spacial: you want to maintain the "layout"
#of the image as you process.  A CNN also performs some filtering
#throughout its processing, which simplifies the image to something
#the computer can more easily understand.  It makes it more clear
#to the computer while simultaneously making it more convoluted
#for humans.
#A CNN is made up of 4 main parts:
#   Convolution layers: these layers place a filter over subset (array) of the image pixels and feeds that subset into a
#       new matrix to be fed into the next layer.  This compresses the data from the subset into a new array.
#   Pooling layer: these layers shrink the sample size of your feature map and makes processing faster and more
#       manageable.  The output is a pooled feature map.  There are two ways to do this:
#           - Max pooling: takes the max input of a particular feature from the convolution layer
#           - Average pooling: takes the average input of a feature from the convolution layer
#
#   Together, the convolution and pooling layers give you feature extraction based on whatever mathematical magic
#   the computer does behind the scenes.  We feed in images, the convolution layers filter and compress the image into
#   data segments to be fed to the pooling layers.  The pooling layers then identify significant or identifying features
#   based on the input from the convolution layer.
#
#   ReLu (rectified linear unit) layer: ensures that the data moves non-linearly through the network and maintains
#       the dimensionality we want.
#   Fully connected layer: this is what allows you to get classification information out of your network.  To get to
#       this layer, we need to flatten the data into a linear format so classifications can be processed.
#
#Now that we have an idea of what a CNN does and how it works, it's time to build the CNN function
#Build a function for the cnn using TensorFlow
def cnn_model(features, labels, mode):
    input_layers = tf.reshape(features['x'], [-1, 50, 50, 3])

    #Set up our convolution layers.  Remember, these are the layers that filter and compress the images into
    #smaller, more informationally dense chunks that the computer will be able to better understand.
    #Convolution Layer 1
    convolution1 = tf.layers.conv2d(
        inputs=input_layers,
        filters=50,
        kernel_size=[7, 7],
        padding='same',
        activation=tf.nn.relu
    )
    #Convolution Layer 2 - takes the information from the Convolution Layer 1 as the inputs
    convolution2 = tf.layers.conv2d(
        inputs=convolution1,
        filters=50,
        kernel_size=[7, 7],
        padding='same',
        activation=tf.nn.relu
    )
    #Convolution Layer 3 - takes the information from the Convolution Layer 2 as the inputs
    convolution3 = tf.layers.conv2d(
        inputs=convolution2,
        filters=50,
        kernel_size=[7, 7],
        padding='same',
        activation=tf.nn.relu
    )

    #Pooling Layer 1 - from the convolutional layers above that culminate in convolution3, we are going to use max
    #pooling to shrink the sample size and move towards feature extraction.
    pool1 = tf.layers.max_pooling2d(
        inputs=convolution3,
        pool_size=[2, 2],
        strides=2
    )

    #Feed the results of the first pooling layer through another convolution layer
    convolution4 = tf.layers.conv2d(
        inputs=pool1,
        filters=5,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )


    pool2 = tf.layers.max_pooling2d(
        inputs=convolution4,
        pool_size=[2, 2],
        strides=2,
        padding='same'
    )

    pool2_flatten = tf.layers.flatten(pool2)

    #Now, feed the data into fully connected layers.  We use TensorFlow dense layers here.
    #Fully-connected layer 1
    fc1 = tf.layers.dense(
        inputs=pool2_flatten,
        units=2000,
        activation=tf.nn.relu
    )
    #Fully-connected layer 2 takes the fully-connected layer 1 as the input
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=1000,
        activation=tf.nn.relu
    )
    #Fully-connected layer 3 takes fully-connected layer 2 as input
    fc3=tf.layers.dense(
        inputs=fc2,
        units=500,
        activation=tf.nn.relu
    )
    #Final fully-connected layer that takes in the third fully-connected layer as input
    logits = tf.layers.dense(
        inputs=fc3,
        units=2
    )

    #Get our predictions from our model.  You can think of the "model" we've built above as the logits results from
    #the last fully-connected layer.
    predictions = {
        'classes':tf.argmax(input=logits, axis=1),
        'probabilities':tf.nn.softmax(logits, name='softmax_tensor')
    }

    #If we are using the function to predict, return the predictions we just gathered
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    #If we are using the function to train the model, run optimization on the model to minimize loss and return the
    #performance estimates
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Otherwise, the other option is using the function to evaluate the performance of our model.  We return the
    #performance estimates to do that.
    eval_metric_op = {'accuracy':tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_op)

#store the infected images
store_images_from_csv('infected', "../malaria/cell_images/cell_images/Parasitized/", 1)

#repeat the process for uninfected images
store_images_from_csv('uninfected', "../malaria/cell_images/cell_images/Uninfected/", 0)

#combine all cell images into one array
cells = np.array(data)

#combine all labels into one array
labels = np.array(labels)

#save cells and labels to hard files
np.save('Cells', cells)
np.save('Labels', labels)

#double check our work to make sure the data exist
#The printed statement should be:
#Cells : (96453, 50, 50, 3) | labels : (96453,)
#print('Cells : {} | labels : {}'.format(cells.shape, labels.shape))

#let's look at some of the cells - this shows 50 images
plt.figure(1, figsize=(15, 9))
n = 0
for i in range(50):
    n += 1
    r = np.random.randint(0, cells.shape[0], 1)
    plt.subplot(7, 8, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.imshow(cells[r[0]])
    #look at the label for the image, and label it as Infected or Uninfected based on the label
    plt.title('{} : {}'.format('Infected' if labels[r[0]] == 1 else 'Uninfected', labels[r[0]]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#now let's look at an infected and an uninfected cell, side by side
plt.figure(1, figsize=(15, 7))
#subplot: 1 pane, 2 images, set the following image as the first
plt.subplot(1, 2, 1)
plt.imshow(cells[0])
plt.title('Infected Cell')
plt.xticks([])
plt.yticks([])

#subplot: 1 pane, 2 images, set the following as the second image
plt.subplot(1, 2, 2)
plt.imshow(cells[60000])
plt.title('Uninfected Cell')
plt.xticks([])
plt.yticks([])
plt.show()

#Moving on
#Shuffle the images
n = np.arange(cells.shape[0])
np.random.shuffle(n)
cells = cells[n]
labels = labels[n]

#Set up the splits for training, evaluating, and testing
cells = cells.astype(np.float32)
labels = labels.astype(np.int32)
cells = cells/255

#Use sklearn.model_selection.train_test_split to get subsets of the data
#for training data, take 20% of the dataset
train_x, x, train_y, y = train_test_split(cells, labels, test_size=0.2, random_state=111)
eval_x, test_x, eval_y, test_y = train_test_split(x, y, test_size=0.5, random_state=111)

#Sanity Check: verify the distributions are roughly equal in each set
plt.figure(1, figsize=(15, 5))
n = 0
for z, j in zip([train_y, eval_y, test_y] , ['train labels', 'eval labels', 'test labels']):
    n += 1
    plt.subplot(1, 3, n)
    sns.countplot(x = z)
    plt.title(j)
plt.show()
#Show how many images are in each set
#Result statement should be:
#train data shape (77162, 50, 50, 3) | eval data shape (9645, 50, 50, 3) | test data shape (9646, 50, 50, 3)
print('train data shape {} | eval data shape {} | test data shape {}'.format(train_x.shape, eval_x.shape, test_x.shape))

#Clear the default graph stack and reset the global default graph (from TensorFlow docs)
tf.reset_default_graph()

#Initialize the model through TensorFlow as malaria_detector
malaria_dectector = tf.estimator.Estimator(model_fn=cnn_model, model_dir='/tmp/modelchpt')

#Let's log some information about the model
tensors_to_log = {'probabilities':'softmax_tensor'}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

#Set up for training the model
train_input = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_x},
    y=train_y,
    batch_size=100,
    num_epochs=None,
    shuffle=True
)
#Train the model
malaria_dectector.train(input_fn=train_input, steps=1, hooks=[logging_hook])
malaria_dectector.train(input_fn=train_input, steps=1000)

#Set up to evaluate the model
eval_input = tf.estimator.inputs.numpy_input_fn(
    x={'x': eval_x},
    y=eval_y,
    batch_size=100,
    num_epochs=None,
    shuffle=False
)
#Evaluate the model
eval_results = malaria_dectector.evaluate(input_fn=eval_input)
print(eval_results)

#Set up to use the model for PREDICTIONS!
predict_input = tf.estimator.inputs.numpy_input_fn(
    x={'x': test_x},
    y=test_y,
    num_epochs=1,
    shuffle=False
)
#Use the model to predict
predictions = malaria_dectector.predict(input_fn=predict_input)
classes = [p['classes'] for p in predictions]

#Use sklearn to show us stats on how well we did
print('{}\n{}\n{}'.format(
    confusion_matrix(test_y, classes),
    classification_report(test_y, classes),
    accuracy_score(test_y, classes)))

#Show a sample of the cells, if they were predicted as healthy or infected, and their actual label
plt.figure(1, figsize=(15, 9))
n = 0
for i in range(50):
    n += 1
    r = np.random.randint(0, test_x.shape[0], 1)
    plt.subplot(5, 10, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.imshow(test_x[r[0]])
    plt.title('actual {} : pred {}'.format(test_y[r[0]], classes[r[0]]))
    plt.xticks([]), plt.yticks([])
plt.show()

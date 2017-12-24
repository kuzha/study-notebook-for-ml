from tensorflow.examples.tutorials.mnist import input_data

# mnist stands for modified NIST (national institue of standards and technology)
# NIST contains the raw database of handwritten digits. It has been modified by
# Yanne LeCun for easier preprocessing and formatting
# one_hot is a vector with only one 1 and with 0 for the rest
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf
# initialize input x, weights W and bias b
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# from x to output y, there are two layers: the first layer is a linear operation
# x*W+b; from the second layer to y, there is a mapping function softmax
# softmax is a activation function like sigmoid
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_ is the real value of labeled data; y is the prediction. ood naming...
y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy is a type of formulation for loss function
# reduce_sum is just an element-wise sum for all the elements in the matrix,
# which has a weird name. The tutorial says this formulation may cause unstability
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# use gradient descent algorithm to minize the cost function
# 0.5 means the changing step is 0.5 in the gradient descent algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# launch the model in InteractivSession. not sure InteractivSession means
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# the training process repeats 1000 times, each time we randomly select 100
# batch of (x,y_) for training. This process is called stochastic gradient
# descent. It is a less computation-expensive process then training using all
# the data we have.
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# After training the NN, we get parameters like W and b for the NN
# but there is no comparison between the training prediction and the real labels
# in the curve fitting process, we normally have a very high fitting rate for the
# traing data sets. Then we use the test data to evaluate the model and normally
# the accuracy will drop in the validation process.

# evaluating the prediction results y with original label y_
# the argmax function gives the index of the highest entry in axis 1. This
# function is a bit blurry for its use here.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

import os
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import lenet5_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"

gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)

def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, lenet5_inference.IMAGE_SIZE, lenet5_inference.IMAGE_SIZE,
      lenet5_inference.NUM_CHANNELS], name='x-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    y_ = tf.placeholder(tf.float32, [None, lenet5_inference.OUTPUT_NODE], name='y-input')
    
    y = lenet5_inference.inference(x, train, regularizer)
    global_step = tf.Variable(0, trainable = False)  #迭代轮数变量控制衰减率， 不像权重不可训练

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.device('/gpu:0'):
            tf.global_variables_initializer().run()
            for i in range(TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                reshaped_xs = np.reshape(xs, [BATCH_SIZE, lenet5_inference.IMAGE_SIZE, lenet5_inference.IMAGE_SIZE,
                    lenet5_inference.NUM_CHANNELS])
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

                if i % 1000 == 0:
                    print ("After %d training step(s), loss on training " "batch is %g." % (step, loss_value))
                    saver.save( sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main (argv=None):
    mnist = input_data.read_data_sets("/path/to/mnist_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

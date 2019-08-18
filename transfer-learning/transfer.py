import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as InceptionV3

INPUT_DATA = '/home/leon/deeplearning/tensorflow/transfer-learning/flower_processed_data.npy'
TRAIN_FILE = '/home/leon/deeplearning/tensorflow/transfer-learning/save_model/'
CKPT_FILE = '/home/leon/deeplearning/tensorflow/transfer-learning/inception_v3.ckpt'

LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32 
N_CLASSES = 5

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'

def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    
    variables_to_restore = []
    for var in slim.get_model_variables():
        print("--------------------------------------------------------------")
        print("The variables_to_restore are:", var)
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(argv=None):
    processed_data = np.load(INPUT_DATA, allow_pickle=True)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_iamges = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print("%d training examples, %d validation exampoles and %d"
        "testing examples." % (n_training_example, len(validation_labels), len(testing_labels)))

    images = tf.compat.v1.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.compat.v1.placeholder(tf.int64, [None], name='labels')

    with slim.arg_scope(InceptionV3.inception_v3_arg_scope()):
        logits, _ = InceptionV3.inception_v3(images, num_classes=N_CLASSES)
    
    trainable_variables = get_trainable_variables()
    tf.compat.v1.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    train_step = tf.compat.v1.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.compat.v1.losses.get_total_loss())

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), 
        ignore_missing_vars=True)

    saver = tf.compat.v1.train.Saver()
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.compat.v1.Session(config=config) as sess:
        
        # sess = tf.Session(config=config)
        with tf.device('/gpu:0'):
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            print('Loading tuned variables from %s' % CKPT_FILE)
            load_fn(sess)
            start = 0
            end = BATCH
            for i in range(STEPS):
                sess.run(train_step, feed_dict={images: training_images[start:end], labels: training_labels[start:end]})

                if i % 60 == 0 or i + 1 == STEPS:
                    saver.save(sess, TRAIN_FILE, global_step=i)
                    validation_accuracy = sess.run(evaluation_step, feed_dict={images: validation_iamges, 
                        labels: validation_labels })
                    print('Step %d: Validation accuracy = %.lf%%' % (i, validation_accuracy* 100.0))
            
                start = end
                if start == n_training_example:
                    start = 0
                end = start + BATCH
                if end > n_training_example:
                    end = n_training_example
            test_accuracy = sess.run(evaluation_step, feed_dict= {
                images: testing_images, labels: testing_labels})
            print('Final test accuracy = %.lf%%' % (test_accuracy* 100.0))

if __name__ == '__main__':
    tf.compat.v1.app.run()





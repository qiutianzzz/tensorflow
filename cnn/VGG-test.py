##########################
### RELOAD & TEST
##########################

import tensorflow as tf
import numpy as np


with tf.Session(graph=g) as sess:
    saver.restore(sess, save_path='./convnet-vgg16.ckpt')
    
    # test_acc = sess.run('accuracy:0', feed_dict={'features:0': X_test,
    #                                              'targets:0': y_test})
    # ---------------------------------------
    # workaround for GPUs with <= 4 Gb memory
    n_predictions, n_correct = 0, 0
    indices = np.arange(y_test.shape[0])
    chunksize = 500
    for start_idx in range(0, indices.shape[0] - chunksize + 1, chunksize):
        index_slice = indices[start_idx:start_idx + chunksize]
        p = sess.run('correct_predictions:0', 
                     feed_dict={'features:0': X_test[index_slice],
                                'targets:0': y_test[index_slice]})
        n_correct += np.sum(p)
        n_predictions += p.shape[0]
    test_acc = n_correct / n_predictions
    # ---------------------------------------

    print('Test ACC: %.3f' % test_acc)
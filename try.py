from __future__ import division, print_function

import cv2
import numpy as np
import tensorflow as tf
import args
import logging
from tqdm import trange


train_dataset = tf.data.TextLineDataset(args.train_file)
print('Direct reading'  train_dataset)
train_dataset = train_dataset.shuffle(args.train_img_cnt)
print('After shuffle'  train_dataset)
train_dataset = train_dataset.batch(args.batch_size)
print('After shuffle'  train_dataset)

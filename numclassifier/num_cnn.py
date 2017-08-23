# -*- coding: utf-8 -*-

"""
    MNIST Classification with TensorFlow on Django
    Copyright (C) 2017  Angel Caban

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""

    input_reshape = tf.reshape(features, [-1, 28, 28, 1])
    input_layer = tf.cast(input_reshape, tf.float32, name="input")

    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat,
                            units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=(mode == learn.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                                   depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                               logits=logits)

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                   global_step=tf.contrib.framework.get_global_step(),
                                                   learning_rate=0.001,
                                                   optimizer="SGD")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return model_fn_lib.ModelFnOps(mode=mode,
                                   predictions=predictions,
                                   loss=loss,
                                   train_op=train_op)

def concat(first, second):
    if len(first.shape) > 1:
        shape = (first.shape[0] + 1, first.shape[1])
    else:
        shape = (first.shape[0] + 1)
    ret = np.concatenate((first.flat, second.flat))
    return ret.reshape(shape)


def predict(pixels):
    estimator = learn.Estimator(model_fn=cnn_model_fn,
                                model_dir="/home/angel/mnist_convnet_model")
    mnist_classifier = learn.SKCompat(estimator)
    
    return mnist_classifier.predict(x=pixels, batch_size=1)


def run_learn(pixels, label):
    mnist = learn.datasets.load_dataset("mnist")

    train_data = concat(mnist.train.images, pixels)
    train_labels = concat(mnist.train.labels, label)

    eval_data = concat(mnist.test.images, pixels)
    eval_labels = concat(mnist.test.labels, label)

    estimator = learn.Estimator(model_fn=cnn_model_fn,
                                model_dir="/home/angel/mnist_convnet_model")
    mnist_classifier = learn.SKCompat(estimator)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=500)

    mnist_classifier.fit(x=train_data,
                         y=train_labels,
                         batch_size=128,
                         steps=10000,
                         monitors=[logging_hook])

    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy,
                                     prediction_key="classes")
    }

    eval_results = mnist_classifier.score(x=eval_data,
                                          y=eval_labels,
                                          metrics=metrics)
    print ("\n", eval_results, "\n")

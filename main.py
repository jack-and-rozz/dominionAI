# coding: utf-8
#coding: utf-8
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys, io, os, codecs, time, itertools, math, random
from logging import FileHandler

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import nn_model

# my sources
from public import config
from public import utils


# global(saved) config
tf.app.flags.DEFINE_string("source_data_dir", "dataset/source", "Data directory")
tf.app.flags.DEFINE_string("train_file", "", "Train file")
tf.app.flags.DEFINE_string("valid_file", "", "Valid file")
tf.app.flags.DEFINE_string("test_file", "", "Test file")

tf.app.flags.DEFINE_string("train_dir", "models/tmp", "Training directory.")
tf.app.flags.DEFINE_float("keep_prob", 1.0,
                          "the keeping probability of active neurons in dropout")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much. This paramerters is not used when using AdamOptimizer.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("max_step", 3000,
                            "Limit on the training step")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", None, "Size of embeddings")

tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_interruption", 0, 
                            "How many training steps to interrupt training for decoding and selection test.")
tf.app.flags.DEFINE_integer("max_to_keep", 5, "Number of checkpoints to be kept")


## temporal flags (not saved in config)
tf.app.flags.DEFINE_string("log_file", None, "Log File")
tf.app.flags.DEFINE_string("mode", "", "")


FLAGS = tf.app.flags.FLAGS
TMP_FLAGS = ['mode', 'log_file']
if FLAGS.log_file: 
  logger = utils.logManager(handler=FileHandler(FLAGS.train_dir + '/' + FLAGS.log_file))
else:
  logger = utils.logManager()

def create_dir():
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)
  if not os.path.exists(FLAGS.train_dir + '/checkpoints'):
    os.makedirs(FLAGS.train_dir + '/checkpoints')
  if not os.path.exists(FLAGS.train_dir + '/tests'):
    os.makedirs(FLAGS.train_dir + '/tests')
  if not os.path.exists(FLAGS.train_dir + '/summaries'):
    os.makedirs(FLAGS.train_dir + '/summaries')

def save_config():
  flags_dir = FLAGS.__dict__['__flags']
  with open(FLAGS.train_dir + '/config', 'w') as f:
    for k,v in flags_dir.items():
      if not k in TMP_FLAGS:
        f.write('%s=%s\n' % (k, str(v)))

def create_model(sess, n_feature, n_target):
  model = nn_model.DominionBaseModel(FLAGS, 
                                     n_feature=n_feature,
                                     n_target=n_target,
                                     mode="train")
  return model

  
def train_random():
  cardlist = data_utils.read_cardlist()
  #_, train_data = data_utils.read_log(FLAGS.source_data_dir + '/' + FLAGS.train_file)
  task_name, valid_data = data_utils.read_log(FLAGS.source_data_dir + '/' + FLAGS.valid_file)
  valid_batch = data_utils.BatchManager(task_name, valid_data)
  #data_utils.explain_state(valid_data[0])
  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    n_feature = valid_batch.n_feature
    n_target = valid_batch.n_target
    model = create_model(sess, n_feature, n_target)
    print model

  pass

def test():
    pass

def main(_):
    create_dir()
    save_config()
    if FLAGS.mode == '--train_random':
          logger.info("[ TRAIN RANDOM ]")
          train_random()
    elif FLAGS.mode == '--test':
          logger.info("[ TEST ]")
          test()
    pass

if __name__ == "__main__":
  tf.app.run()

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
from tensorflow.python.platform import gfile

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
tf.app.flags.DEFINE_string("task_name", "Gain", "")
tf.app.flags.DEFINE_float("keep_prob", 1.0,
                          "the keeping probability of active neurons in dropout")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
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
  #logger = utils.logManager()
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

@utils.timewatch(logger)
def create_model(sess, n_feature, n_target):

  model = nn_model.GainModel(FLAGS, 
                             n_feature=n_feature,
                             n_target=n_target,
                             mode="train")
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + '/checkpoints')
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_to_keep)
    saver.restore(sess, ckpt.model_checkpoint_path)
  else:
    logger.info("Created model with fresh parameters.")
    tf.initialize_all_variables().run()

  return model


@utils.timewatch(logger)  
def gain_test():
  FLAGS.batch_size = 1

  cardlist = data_utils.read_cardlist()
  _, test_data = data_utils.read_log(FLAGS.source_data_dir + '/' + FLAGS.test_file)
  test_batch = data_utils.BatchManager(test_data)
  n_feature = test_batch.n_feature
  n_target = test_batch.n_target
  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    model = create_model(sess, n_feature, n_target)

    results = []
    for i in xrange(test_batch.size):
      print "--- Test %d --- "% i
      input_data, targets = test_batch.get(FLAGS.batch_size)
      answer = sorted(model.buy(sess, input_data))
      correct_answer = sorted(test_data[i]['answer'])
      while True:
        if 0 in answer and len(answer) > 1:
          answer.remove(0)
        else:
          break
      data_utils.explain_state(input_data[0], answer, correct_answer)
      results.append(answer == correct_answer)
    print 1.0 * results.count(True) / len(results)

@utils.timewatch(logger)
def train_random():
  t = time.time()
  _, train_data = data_utils.read_log(FLAGS.source_data_dir + '/' + FLAGS.train_file)
  _, valid_data = data_utils.read_log(FLAGS.source_data_dir + '/' + FLAGS.valid_file)
  logger.info("Reading data : %f sec" % (time.time() - t))
  train_batch = data_utils.BatchManager(train_data)
  valid_batch = data_utils.BatchManager(valid_data)

  with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    n_feature = train_batch.n_feature
    n_target = train_batch.n_target
    model = create_model(sess, n_feature, n_target)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir + '/summaries', 
                                            graph=sess.graph)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_to_keep)
    # 以降、opsの定義禁止
    t_ave_loss, step_time = 0.0, 0.0
    logits_op, loss_op, train_op = model.logits_op, model.loss_op, model.train_op
    summary_op = tf.merge_all_summaries()
    logger.info("Start training")
    while True:
      input_data, targets = train_batch.get(FLAGS.batch_size)
      
      train_feed = {
        model.input_data : input_data,
        model.targets : targets,
      }
      #print input_data, targets
      t = time.time()
      t_logits, t_loss, _ = sess.run([logits_op, loss_op, train_op], 
                                     train_feed)
      step_time += (time.time() - t) / FLAGS.steps_per_checkpoint
      step = model.global_step.eval()
      t_ave_loss += t_loss / FLAGS.steps_per_checkpoint if step != 0 else t_loss 

      if step % (FLAGS.steps_per_checkpoint) == 0:
        t_ave_ppx = math.exp(t_ave_loss) if t_ave_loss < 300 else float('inf')

        n_eval = int(valid_batch.size / FLAGS.batch_size)
        v_ave_ppx = 0.0
        for _ in xrange(n_eval):
          input_data, targets = valid_batch.get(FLAGS.batch_size)
          valid_feed = {
            model.input_data : input_data,
            model.targets : targets,
          }
          
          v_logits, v_loss = sess.run([logits_op, loss_op], valid_feed)
          v_ave_ppx += math.exp(v_loss)/(n_eval) if v_loss < 300 else float('inf')
        logger.info("global step %d step-time %.4f" % (step,
                                                       step_time))
        logger.info("Train ppx %.4f" % t_ave_ppx)
        logger.info("Valid ppx %.4f" % v_ave_ppx)
        t_ave_loss, step_time = 0.0, 0.0

        #summary_str = sess.run(summary_op, feed_dict=train_feed)
        #summary_writer.add_summary(summary_str, step)
        # Save checkpoint
        checkpoint_path = os.path.join(FLAGS.train_dir, "checkpoints/model.ckpt")
        saver.save(sess, checkpoint_path, global_step=model.global_step)
      if step >= FLAGS.max_step:
        break;
      
def main(_):
    create_dir()
    save_config()
    if FLAGS.mode == '--train_random':
          logger.info("[ TRAIN RANDOM ]")
          train_random()
    elif FLAGS.mode == '--test':
          logger.info("[ TEST ]")
          gain_test()
    pass

if __name__ == "__main__":
  tf.app.run()

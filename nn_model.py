#coding: utf-8
import copy
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import numpy as np
import data_utils as du

class DominionBaseModel(object):
  def __init__(self, flags, n_feature, n_target, mode="train"):
    self.hidden_size = flags.hidden_size
    self.num_layers = flags.num_layers
    self.max_gradient_norm = flags.max_gradient_norm
    self.batch_size = batch_size = flags.batch_size
    self.learning_rate = flags.learning_rate
    self.mode = mode
    self.n_feature = n_feature
    self.n_target = n_target
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.max_to_keep = flags.max_to_keep

    self.input_data = tf.placeholder(tf.float32, [batch_size, n_feature])
    self.targets = tf.placeholder(tf.float32, [batch_size, n_target])

    self.logits_op = self.inference()
    self.loss_op = self.loss(self.logits_op)
    self.train_op = self.optimize(self.loss_op)
    self.probs_op = tf.nn.softmax(self.logits_op)

  def __str__(self):
    return str(self.__dict__)

  def optimize(self, loss):
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    #optimizer  tf.trin.GradientDescentOptimizer(step_size)
    train_op = optimizer.minimize(loss, global_step=self.global_step)
    return train_op

  def loss(self, logits):
    labels = self.targets
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                            labels,
                                                            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


  def inference(self):
    hidden = []
    with vs.variable_scope('hidden0') as scope:
      w = tf.get_variable("weights", [self.n_feature, self.hidden_size])
      b = tf.get_variable("bias", [self.hidden_size])
      h = tf.nn.sigmoid(tf.matmul(self.input_data, w) + b)
      hidden.append(h)

    for i in xrange(1, self.num_layers):
      with vs.variable_scope('hidden%d' % i) as scope:
        w = tf.get_variable("weights", [self.hidden_size, self.hidden_size])
        b = tf.get_variable("bias", [self.hidden_size])
        h = tf.nn.sigmoid(tf.matmul(hidden[i-1], w) + b)
        hidden.append(h)

    with vs.variable_scope('projection') as scope:
      w = tf.get_variable("weights", [self.hidden_size, self.n_target])
      b = tf.get_variable("bias", [self.n_target])
      logits = tf.nn.sigmoid(tf.matmul(hidden[-1], w) + b)
    return logits

class GainModel(DominionBaseModel):
  def buy(self, sess, input_data):
    #とりあえずバッチ実行でテストは考えない
    test_feed = {
      self.input_data : input_data,
    }
    
    #t_logits = sess.run(self.logits_op, test_feed)
    #target = self.select_validly(input_data[0], t_logits[0])
    t_probs = sess.run(self.probs_op, test_feed)
    target = self.select_validly(input_data[0], t_probs[0])
    next_input, continuable = self.update_state(input_data[0], target)
    res = [target]
    if continuable:
      res.extend(self.buy(sess, next_input))
    return res

  def select_validly(self, input_data, logits):
    supply_ids = du.included_supplies(input_data)
    game_info = du.get_game_info(input_data, supply_ids)
    supply_logits = [logits[i] for i in supply_ids]
    coin = game_info['coin']
    cost = game_info['cost']
    supply = game_info['supply']
    # サプライに無い or コストが足りないものはlogitを0に 
    not_buyables = [i for i, _id in enumerate(supply_ids) 
                    if (coin < cost[i] or (_id != 0 and supply[i] == 0))]
    for i in not_buyables:
      supply_logits[i] = 0
    supply_logits /= sum(supply_logits)
    
    probs = [(game_info['name'][i], supply_logits[i]) for i in xrange(len(supply_ids)) if supply_logits[i] >= 0.01 ]
    probs = sorted(probs, key=lambda x:-x[1])
    print "<Buy Probability>"
    print ", ".join(["%s(%.4f)" % (name, prob) for (name, prob) in probs])
    return supply_ids[np.argmax(supply_logits)]

  def update_state(self, input_data, target):
    #<other_features> = ['coin', 'buy', 'minusCost', 'turn']
    #<stateVec>	deck,hand,discard,field,enemy,supply,isIncludedSupply (cardmax*7次元)
    next_input = copy.deepcopy(input_data)
    supply_ids = du.included_supplies(input_data)
    game_info = du.get_game_info(input_data, supply_ids)
    cost = game_info['cost'][supply_ids.index(target)]
    idx = du.get_feature_start_idx()
    next_input[idx['coin']] -= cost
    next_input[idx['buy']] -= 1
    if target != 0:
      next_input[idx['discard'] + target] += 1
      next_input[idx['supply'] + target] -= 1

    supply = game_info['supply']
    continuable = True
    if next_input[idx['buy']] == 0:
      continuable = False
    elif len(supply) - np.count_nonzero(supply) >= 3:
      continuable = False
    return [next_input], continuable
  def buy_beamsearch(self, sess, input_data, targets):
    pass

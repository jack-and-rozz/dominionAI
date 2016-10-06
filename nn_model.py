#coding: utf-8
import tensorflow as tf

class DominionBaseModel(object):
  def __init__(self, flags, n_feature, n_target, mode="train"):
    self.hidden_size = flags.hidden_size
    self.num_layers = flags.num_layers
    self.max_gradient_norm = flags.max_gradient_norm
    self.batch_size = flags.batch_size
    self.mode = mode
    self.n_feature = n_feature
    self.n_target =n_target

    #self.inputs = tf.placeholder(tf.float32, shape=[None], name='inputs')
    #self.outputs = tf.placeholder(tf.float32, shape=[None], name='outputs')
    self.inputs = tf.placeholder(tf.float32, shape=[self.n_feature], name='inputs')
    self.outputs = tf.placeholder(tf.float32, shape=[self.n_target], name='outputs')

  def __str__(self):
    return str(self.__dict__)

  def inference(self):
    hidden = []
    with tf.name_scope('hidden0') as scope:
      w = tf.get_variable("weights", [self.n_feature, self.hidden_size])
      b = tf.get_variable("bias", [self.hidden_size])
      h = tf.nn.sigmoid(tf.matmul(self.inputs, w) + b)
      hidden.append(h)

    for i in xrange(1, self.num_layers):
      with tf.name_scope('hidden%d' % i) as scope:
        w = tf.get_variable("weights", [self.hidden_size, self.hidden_size])
        b = tf.get_variable("bias", [self.hidden_size])
        h = tf.nn.sigmoid(tf.matmul(hidden[i-1], w) + b)
        hidden.append(h)

    with tf.name_scope('projection') as scope:
      w = tf.get_variable("weights", [self.hidden_size, self.n_target])
      b = tf.get_variable("bias", [self.n_target])
      y = tf.nn.sigmoid(tf.matmul(hidden[-1], w) + b)
    return y

#class GainModel(DominionBaseModel):
#    def __init__(self, flags, n_target, mode="train"):
#        super(GainModel, self).__init__(flags, mode)

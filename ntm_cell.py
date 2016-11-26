import numpy as np
import tensorflow as tf
from ops import *


class NTMCell():

  def __init__(self, num_units, memory_shape, activation=tf.tanh, batch_size=1):
    self._num_units = num_units
    self._activation = activation
    self._batch_size = batch_size
    self.memory_shape = memory_shape
    self.mem_size = memory_shape[0]
    self.mem_dim = memory_shape[1]


  def init_state(self, reuse=None):

    with tf.variable_scope("init_cell"):

      start_vec = tf.constant([[0.0]], dtype=tf.float32)

      memory = tf.tanh(linear(start_vec, self.mem_size*self.mem_dim, name="memory_init", reuse=reuse))
      memory = tf.reshape(memory, shape=[self.mem_size, self.mem_dim])

      w_read = linear(start_vec, self._batch_size*self.mem_size, name="w_read_init", reuse=reuse)
      w_read = tf.reshape(w_read, shape=[self._batch_size, self.mem_size])

      w_write = linear(start_vec, self._batch_size*self.mem_size, name="w_write_init", reuse=reuse)
      w_write = tf.reshape(w_write, shape=[self._batch_size, self.mem_size])

      prev_h = tf.tanh(linear(start_vec, self._batch_size*self._num_units, name="prev_h_init", reuse=reuse))
      prev_h = tf.reshape(prev_h, shape=[self._batch_size, self._num_units])


    self.state = {
      'M': memory,
      'w_read': w_read,
      'w_write': w_write,
      'prev_h': prev_h
    }
    

  def _read_from_memory(self):
    w_read = self._get_read_weights()
    M = self.state['M']
    return tf.matmul(w_read, M)

  
  def _write_to_memory(self):
    # write weights, erase row vector(e), and add row vector(a)
    w_write, e, a = self._get_write_weights() 
    M_tilde = self.state['M'] * (tf.ones(self.memory_shape) - tf.transpose(tf.matmul(e, w_write)))

    self.state['M'] = M_tilde + tf.matmul(tf.transpose(w_write), a)


  def _get_read_weights(self):
    return self._get_weights('READ')

  def _get_write_weights(self):
    return self._get_weights('WRITE')


  def _get_weights(self, htype):
    if htype == 'READ':
      head = self._get_read_head()
    elif htype == 'WRITE':
      head = self._get_write_head()

    k, beta, g, s, gamma = head['k'], head['beta'], head['g'], head['s'], head['gamma']
    M = self.state['M']

    # 3.3.1 Focusing by Content
    K = cosine_similarity(k, M)
    w_c = tf.nn.softmax(beta*K + 1e-7)

    # 3.3.2 Focusing by Location
    w_g = tf.mul(g, w_c) + tf.mul((tf.ones(shape=g.get_shape().as_list()) - g), self.state['w_read'])
    w_tilde = circular_convolution(w_g, s)

    w = tf.nn.softmax(tf.pow(w_tilde, gamma) + 1e-7)
    
    if htype == 'READ':
      self.state['w_read'] = w
      return w
    else:
      self.state['w_write'] = w
      return w, head['erase'], head['add']


  def _get_read_head(self):
    return self._get_head('READ')

  def _get_write_head(self):
    return self._get_head('WRITE')


  def _get_head(self, htype):
    scope = 'read' if htype == 'READ' else 'write'

    out = self.state['prev_h']
    
    with tf.variable_scope(scope):

      with tf.variable_scope("k"):
        k = tf.nn.relu(linear(out, self.mem_dim, "k"))

      with tf.variable_scope("beta"):
        beta = tf.nn.relu(linear(out, 1, "beta"))

      with tf.variable_scope("g"):
        g = tf.sigmoid(linear(out, 1, "g"))

      with tf.variable_scope("s"):
        s = tf.nn.softmax(linear(out, 3, "s") + 1e-7)

      with tf.variable_scope("gamma"):
        gamma = tf.nn.relu(linear(out, 1, "gamma")) + 1.0

      if htype == 'WRITE':
        with tf.variable_scope("erase"):
          erase = tf.sigmoid(linear(out, self.mem_dim, "erase"))
          erase = tf.transpose(erase)
        
        with tf.variable_scope("add"):
          add = tf.nn.relu(linear(out, self.mem_dim, "add"))
          tf.add = tf.transpose(add)

    head = {
      'k': k,
      'beta': beta,
      'g': g,
      's': s,
      'gamma': gamma,
    }

    if htype == 'WRITE':
      head.update({'erase': erase, 'add': add})

    return head


  def _controller(self, x_t, read_t, reuse=None):

    x = tf.concat(1, [x_t, self.state['prev_h'], self.state['w_read']])

    with tf.variable_scope("z"):
      z = linear(x, self._num_units, name="z") 
      z = tf.sigmoid(z)

    with tf.variable_scope("c"):
      c = linear(x, self._num_units, name="c") 
      c = tf.sigmoid(c)

    with tf.variable_scope("h_tilde"):
      in_ = tf.concat(1, [c*self.state['prev_h'], x_t])
      h_tilde = tf.tanh(linear(in_, self._num_units, name="h_tilde"))

    next_h = (tf.ones(shape=z.get_shape().as_list()) - z) * self.state['prev_h'] + z * h_tilde
    self.state['prev_h'] = next_h
    
    return next_h, next_h


  def __call__(self, inputs):
    read_t = self._read_from_memory()
    out_t, next_h = self._controller(inputs, read_t)
    self._write_to_memory()
    return out_t, next_h

  
    

  

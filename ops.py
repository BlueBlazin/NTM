import tensorflow as tf
import math

def linear(x, output_size, name=None, add_bias=True, bias=0.0, reuse=None):
  shape = x.get_shape().as_list()
  name = "generic" if name == None else name

  with tf.variable_scope("weight", reuse=reuse):
    W = tf.get_variable(name=name, 
      shape=[shape[1], output_size],
      initializer=tf.truncated_normal_initializer(stddev=1e-4))

  if not add_bias:
    return tf.matmul(x, W)

  with tf.variable_scope("bias", reuse=reuse):
    b = tf.get_variable(name=name,
      shape=[output_size],
      initializer=tf.constant_initializer(bias))

  return tf.matmul(x, W) + b



def circular_convolution(wg, s):
  wg = tf.transpose(wg)
  s  = tf.transpose(s)
  N = int(wg.get_shape()[0])
  kernel_size = int(s.get_shape()[0])

  out = []
  for i in xrange(N):
    indices = [(i-j) % kernel_size for j in xrange(N)]
    s_ = tf.gather(s, indices)
    out.append(tf.reshape(tf.reduce_sum(s_ * wg, 0), [1, -1]))

  return tf.transpose(tf.concat(0, out))
  


def cosine_similarity(u, v, epsilon=1e-6):
  norm_u = tf.sqrt(tf.reduce_sum(tf.square(u), 1) + epsilon)
  norm_v = tf.sqrt(tf.reduce_sum(tf.square(v), 1) + epsilon)

  u = u / tf.reshape(norm_u, shape=[-1, 1])
  v = v / tf.reshape(norm_v, shape=[-1, 1]) 

  return tf.matmul(u, tf.transpose(v))

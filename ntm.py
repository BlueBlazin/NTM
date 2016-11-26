import numpy as np
import tensorflow as tf
from ops import linear


class NTM():

  def __init__(self, cell, num_steps, input_size, init=True):
    self.cell = cell
    self.num_steps = num_steps
    self.input_size = input_size
    if init:
      self.cell.init_state()


  def _get_outputs(self, inputs):
    outputs = []
    for i, x_t in enumerate(inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      h, _ = self.cell(x_t)
      with tf.variable_scope("output"):
        output = linear(h, self.input_size, name="w_out")
      outputs.append(output)
    
    return outputs


  def _get_loss(self, outputs, y):
    losses =\
     [tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(o, y_)) for o, y_ in zip(outputs, y)]
    return sum(losses) / self.num_steps


  def __call__(self, inputs, y=None, train=False):
    self.cell.init_state(reuse=True)
    logits = self._get_outputs(inputs)
    outputs = [tf.sigmoid(o) for o in logits]
    if train:
      loss = self._get_loss(logits, y)
      return outputs, loss
    return outputs

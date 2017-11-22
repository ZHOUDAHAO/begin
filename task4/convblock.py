import tensorflow as tf
from mytf import *

class CB(object):
	def __init__(self, in_cha, out_cha, scope = None):
		self.scope = scope if not scope else scope
		with tf.variable_scope(self.scope):
			self.conv1 = myinit('conv1',[3, in_cha, out_cha])
			self.conv2 = myinit('conv2',[3, out_cha, out_cha])

			self.bias1 = myinit('bias1',[out_cha])
			self.bias2 = myinit('bias2',[out_cha])

	# down sampling means first conv layer has stride 2
	def create_model(self, inputs, down_sample = False):
		with tf.variable_scope(self.scope):
			if down_sample:
				self._conv1 = conv_wrapper(conv1d, inputs, self.conv1, 2)
			else:
				self._conv1 = conv_wrapper(conv1d, inputs, self.conv1, 1)
			self._tbn1 = tf.contrib.layers.batch_norm(self._conv1)
			self._relu1 = tf.nn.relu(self._tbn1)

			self._conv2 = conv_wrapper(conv1d, self._relu1, self.conv2, 1)
			self._tbn2 = tf.contrib.layers.batch_norm(self._conv2)
			self._relu2 = tf.nn.relu(self._tbn2)
			return self._relu2

	def get_weights(self):
		return [self.conv1, self.conv2]

	def get_remain(self):
		return [self.bias1, self.bias2]

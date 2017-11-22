import tensorflow as tf
from mytf import *

class SE(object):
	def __init__(self, out_cha, conv, scope = None):
		self.scope = scope if not scope else scope
		with tf.variable_scope(self.scope):
			self.fc1 = myinit('fc1',[1, 1, out_cha, out_cha])
			self.bias1 = myinit('bias1',[out_cha])
			self.fc2 = myinit('fc2',[1, 1, out_cha, out_cha])
			self.bias2 = myinit('bias2',[out_cha])
			self.conv = conv

	def __call__(self, x, stride):
		res = conv1d(x, self.conv, stride)
		res = tf.expand_dims(res, [2])
		w = res.get_shape().as_list()[-3]
		
		squeeze = tf.nn.avg_pool(res, [1,w,1,1], [1,1,1,1], padding = 'VALID')
		squeeze = tf.transpose(squeeze, [1, 2, 0, 3]) # squeeze: 1*1*batch_size*channel

		fc1 = tf.matmul(squeeze, self.fc1) + self.bias1
		fc2 = tf.matmul(fc1, self.fc2) + self.bias2 # fc2: 1*1*batch_size*channel

		excition = tf.transpose(fc2, [2,0,1,3])
		excition = tf.tile(excition, [1,w,1,1])
		res = tf.multiply(excition, res)
		res = tf.squeeze(res, [2])
		return res

class CB(object):
	def __init__(self, in_cha, out_cha, scope = None):
		self.scope = scope if not scope else scope
		with tf.variable_scope(self.scope):
			self.conv1 = myinit('conv1',[3, in_cha, out_cha])
			self.conv2 = myinit('conv2',[3, out_cha, out_cha])

			self.se1 = SE(out_cha, self.conv1, 'se1')
			self.se2 = SE(out_cha, self.conv2, 'se2')

	# down sampling means first conv layer has stride 2
	def create_model(self, inputs, down_sample = False):
		with tf.variable_scope(self.scope):
			if down_sample:
				self._conv1 = self.se1(inputs, 2)
			else:
				self._conv1 = self.se1(inputs, 1)
			self._tbn1 = tf.contrib.layers.batch_norm(self._conv1)
			self._relu1 = tf.nn.relu(self._tbn1)

			self._conv2 = self.se2(self._relu1, 1)
			self._tbn2 = tf.contrib.layers.batch_norm(self._conv2)
			self._relu2 = tf.nn.relu(self._tbn2)
			return self._relu2

	def get_weights(self):
		return [self.conv1, self.conv2]

import numpy as np
import tensorflow as tf

from auxi import *
from mytf import *
from convblock import *
from Preprocess.AGnews import load_test,alphabet,sample_train
from tf_classify_summary import tf_classify_summary

import time,random

# x is a (h + 4)*(w + 4) matrix
def local(x, W, b, i, j, in_cha_num, out_cha):
	s = sum([x[b, i+1, j+1, in_cha] * W[i, j, in_cha, out_cha] for in_cha in range(in_cha_num) for i in range(3) for j in range(3)])
	return s

def deformable(x, W, offset_conv):
	xshape = x.get_shape().as_list()
	Wshape = W.get_shape().as_list()
	bsize = tf.shape(x)[0]

	resh = xshape[-3] - Wshape[0]
	resw = xshape[-2] - Wshape[1]
	out_cha = Wshape[-1]
	in_cha = Wshape[-2]

	res = tf.zeros([bsize, resh, resw, out_cha], dtype = tf.float32)
	for b in range(bsize):
		for o in range(out_cha):
			for i in range(2, 2 + resh):
				for j in range(2, 2 + resw):
					res[b,i,j,o] = local(x, W, b, i, j, in_cha, out_cha)

# 3 * 3 deformable convolution
def deformable_conv33(x, W, offset_conv):
	padx1 = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
	offset = conv2d(padx1, offset_conv, stride = 1, padding = 'SAME')
	padx2 = tf.pad(padx1, [[0,0],[1,1],[1,1],[0,0]])
	return deformable(padx2, W, offset_conv)

class Config:
	wordvecdim = 16
	n_classes = 4
	in_width = 600 

	alphabet_size = len(alphabet)

	lr = 0.002
	lamb = 0.00003

	train_times = 16000
	mini_bsize = 64
	train_num = 120000
	test_num = 7600

class Model(tf_classify_summary):
	def add_placeholder(self, yshape):	
		self.lookup = tf.placeholder(tf.int32, [None, Config.in_width], name = 'shuffle')
		self.labels = tf.placeholder(tf.float32, yshape, name = 'labels')
		self.lr = tf.placeholder(tf.float32, name = 'lr')

	def sample_labels(self, shuffle, flag):
		if flag == 1: # train
			labels = get_item_by_idx(self.train_labels, shuffle)
		elif flag == 0: # test
			labels = get_item_by_idx(self.test_labels, shuffle)
		else:
			raise Exception('flag should be 1 or 0')
		return labels

	def init_variable(self): 
		self.epoch = 0
		self.alpha_vec = myinit('alpha_vec', [len(alphabet), Config.wordvecdim])

		self.temp_conv = myinit('temp_conv',[3, 16, 64])
		self.cb1 = CB(64, 64, 'cb1')
		self.cb2 = CB(64, 128, 'cb2')
		self.cb3 = CB(128, 256, 'cb3')
		self.cb4 = CB(256, 512, 'cb4')

		with tf.variable_scope('global'):
			self.fc1 = myinit('fc1',[4096, 2048])
			self.bias1 = myinit('bias1',[2048])
			self.fc2 = myinit('fc2',[2048, 2048])
			self.bias2 = myinit('bias2',[2048])
			self.fc3 = myinit('fc3',[2048, Config.n_classes])
			self.bias3 = myinit('bias3',[Config.n_classes])

		all_weights = []
		all_weights.append(self.temp_conv)

		all_weights.extend(self.cb1.get_weights())
		all_weights.extend(self.cb2.get_weights())
		all_weights.extend(self.cb3.get_weights())
		all_weights.extend(self.cb4.get_weights())

		all_weights.append(self.fc1)
		all_weights.append(self.fc2)
		all_weights.append(self.fc3)
		for w in all_weights:
			tf.add_to_collection(tf.GraphKeys.WEIGHTS, w) 
	
	def add_prediction_op(self): 
		self.assign_op = self.alpha_vec[-1].assign(tf.zeros(Config.wordvecdim))
		if self.assign_op is not None:
			with tf.control_dependencies([self.assign_op]):
				self.x = tf.nn.embedding_lookup(self.alpha_vec, self.lookup)
		else:
			self.x = tf.nn.embedding_lookup(self.alpha_vec, self.lookup)
		_temp_conv = conv_wrapper(conv1d, self.x, self.temp_conv, 1)
		_temp_conv = tf.contrib.layers.batch_norm(_temp_conv)

		_cb1 = self.cb1.create_model(_temp_conv, down_sample = True)
		_cb2 = self.cb2.create_model(_cb1, down_sample = True)
		_cb3 = self.cb3.create_model(_cb2, down_sample = True)
		_cb4 = self.cb4.create_model(_cb3, down_sample = True)

		_cb4_T = tf.transpose(_cb4, [0, 2, 1])
		_k_max_pool, _ = tf.nn.top_k(_cb4_T, k = 8)

		_fc_input = tf.reshape(_k_max_pool, [-1, 4096])
		_fc1 = tf.matmul(_fc_input, self.fc1) + self.bias1
		_relu1 = tf.nn.relu(_fc1)
		_fc2 = tf.matmul(_relu1, self.fc2) + self.bias2
		_relu2 = tf.nn.relu(_fc2)
		_fc3 = tf.matmul(_relu2, self.fc3) + self.bias3
		return _fc3
		
	def build(self):
		self.init_variable()

		self.add_placeholder([None, Config.n_classes])
		self.pred = self.add_prediction_op()
		tf.summary.histogram('pred', self.pred)

		self.loss = self.add_loss_op(self.pred, Config)
		tf.summary.scalar('Loss', self.loss)
		self.train_op = self.add_train_op(self.loss)
		
		self.acc = self.add_acc_op(self.pred)
		tf.summary.scalar('acc', self.acc)

	def get_batch_feed(self, a, b, batch_size, flag, lr):
		shuffle = get_shuffle(a, b, batch_size)
		if flag == 1: # train
			lookup_batch, labels_batch = sample_train(shuffle, Config.in_width)
			labels_batch = get_labels(labels_batch, Config.n_classes)
			feed = {self.lookup: lookup_batch, self.labels:labels_batch, self.lr:lr}
		elif flag == 0: # test
			lookup_batch, labels_batch = load_test(Config.in_width)
			labels_batch = get_labels(labels_batch, Config.n_classes)
			feed = {self.lookup: lookup_batch, self.labels:labels_batch, self.lr:lr}
		else:
			raise Exception('flag should be 1 or 0')
		return feed

if __name__ == "__main__":
	np.seterr(all = 'raise')
	m = Model()
	m.train(Config, sgd = True)

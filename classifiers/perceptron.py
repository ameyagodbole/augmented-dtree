import tensorflow as tf
import os
import numpy as np
import pandas as pd
import logging
from classifiers.classifier import Classifier
import time

def timing(f):
	def wrap(*args):
		time1 = time.time()
		ret = f(*args)
		time2 = time.time()
		print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
		return ret
	return wrap


class Perceptron(Classifier):
	"""Implement a single layer Perceptron"""

	def __init__(self, input_dim, output_dim, num_classes, epochs, batch_size):
		"""
		Arguments:
		input_dim:	Dimension of input data
		output_dim:	Dimension of output labels (equal to number of child nodes)
		num_classes: Number of classes in data
		epochs:		Number of training epochs
		batch_size:	Samples per batch (training as well as prediction)
		"""
		super(Perceptron, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_classes = num_classes
		self.epochs = epochs
		self.batch_size = batch_size
		self.built = False
		self.session = None
		self.graph = None
		self.score = 0.0

	def build(self):
		"""
		Build classifier graph
		"""
		logging.debug('Build graph')
		self.graph = tf.Graph()
		with self.graph.as_default():
			with tf.variable_scope('input') as scope:
				self.data = tf.placeholder(tf.float32, [None, self.input_dim], name="Xin")
				# one hot encoded sample vs target class
				self.label = tf.placeholder(tf.float32, [None, self.num_classes], name="Y")
				self.lr = tf.placeholder(tf.float32, [], name="lr")
				global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
			with tf.variable_scope('perceptron') as scope:
				clip_val_min = tf.constant(1e-37, dtype=tf.float32)
				clip_val_max = tf.constant(1, dtype=tf.float32)
				tfinf = tf.constant(np.inf, dtype=tf.float32)
				self.W = tf.get_variable('weight', shape=[self.input_dim, self.output_dim], dtype=tf.float32,
					initializer=tf.contrib.layers.xavier_initializer())
				self.b = tf.get_variable('bias', shape=[1, self.output_dim], dtype=tf.float32,
					initializer=tf.ones_initializer())
				W_norm = tf.clip_by_value(tf.norm(self.W, ord=1, axis=0, keep_dims=True),
				 clip_value_min=clip_val_min, clip_value_max=tfinf)
				# q.shape = [None, output_dim]
				self.q = tf.nn.softmax((tf.matmul(self.data, self.W, name='matmul') + self.b)/W_norm, name='softmax')
			with tf.variable_scope('loss') as scope:
				# n.shape = [num_classes, output_dim]
				n = tf.matmul(tf.matrix_transpose(self.label, name='label_transpose'), self.q, name='n')
				# q.shape = [1, output_dim]
				q_sum = tf.reduce_sum(self.q, axis=0, keep_dims=True, name='q_sum')
				N = tf.reduce_sum(q_sum, name='N')
				# p.shape = [num_classes, output_dim]
				p = tf.divide(n, q_sum, name='p')
				p_clip = tf.clip_by_value(p, clip_value_min=clip_val_min, clip_value_max=clip_val_max)
				# H.shape = [1, output_dim]
				H = -1 * tf.reduce_sum(tf.multiply(p_clip, tf.log(p_clip, name='log_p')), axis=0, keep_dims=True, name='Hj')
				self.loss = tf.reduce_sum(tf.multiply(H, q_sum/N), name='weighted_entropy')
			with tf.variable_scope('opt') as scope:
				self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=global_step)

		self.built = True

	@timing
	def train(self, data_file, balanced_file, child_id):
		"""
		Train on data and return params
		Arguments:
		data_file:	File containing the original data
		balanced_file:	File containing the balanced data
		child_id:	List of child nodes (used for saving split data)
		"""
		if not self.built:
			logging.error("Perceptron: train called before build")
			raise AssertionError("Perceptron: train called before build")
		params = {}
		self.session = tf.Session(graph = self.graph)
		with self.session as sess:
			sess.run(tf.global_variables_initializer())
			df = pd.read_csv(balanced_file)
			lr = np.float32(self.batch_size)/(len(df)*10.)
			for e in range(self.epochs):
				epoch_loss = 0.0
				num_samples = 0
				for batch in self.batch_generator(df):
					_, bloss = sess.run([self.train_op, self.loss], feed_dict={self.data: batch[0], self.label:batch[1], self.lr:lr})
					epoch_loss += bloss*batch[0].shape[0]
					num_samples += batch[0].shape[0]
				if e%10==0:
					logging.info("End of epoch {}".format(e+1))
					logging.info("Average epoch loss : {}".format(epoch_loss/num_samples))

			logging.debug('Running predictions on {} for generating split'.format(data_file))
			preds = []
			dfo = pd.read_csv(data_file)
			for batch in self.batch_generator(dfo, shuffle=False):
				pred = sess.run(self.q, feed_dict={self.data: batch[0], self.label:batch[1]})
				preds += pred.tolist()
			self.split_dataset(data_file, np.asarray(preds), child_id)

			params['W'] = self.W.eval()
			params['b'] = self.b.eval()

		return params

	@timing
	def predict(self, node_id, params, df, child_id):
		"""
		Predicts on dataframe
		Arguments:
		node_id:	ID of node containing the decision_maker
		params:		Dictionary of decision_maker parameters
		data:		DataFrame of test samples.
					NOTE: label column will be ignored. Assumes the indexing o dataframe
						is done using the assigned node i.e. samples reaching current node
						can be accessed by df.ix[self.node_id]
		child_id:	List of child node IDs used to update the index
		"""
		logging.debug('Node {} : Run predict'.format(node_id))
		x = df.ix[node_id, (df.columns!='predicted_label') & (df.columns!='label')].as_matrix()
		Wx = x.dot(params['W'])
		Wxb = Wx + params['b']
		preds = np.argmax(Wxb, 1)
		output = np.asarray(child_id)[preds.astype(np.int32)].tolist()

		as_list = np.asarray(df.index.tolist())
		idx = np.where(as_list==node_id)[0]
		as_list[idx] = output
		df.index = as_list

	def is_label(self, data_file, count_threshold, purity_threshold):
		"""
		Checks if the data should be split or declared a leaf node
		Arguments:
		data_file:	File with data samples
		count_threshold:	Minimum samples needed to consider tree growth
		purity_threshold:	Percentage of most common class for purity-based stoppping of tree growth
		"""
		df = pd.read_csv(data_file)
		if len(df) < count_threshold:
			logging.debug('Decide label node based on count_threshold')
			return True
		counts = np.asarray([len(df[df['label']==c]) for c in range(self.num_classes)]).astype(np.float32)
		if np.float(np.max(counts))/len(df) > purity_threshold:
			logging.debug('Decide label node based on purity')
			return True
		return False

	def max_freq(self, data_file):
		"""
		Get most frequent class
		Arguments:
		data_file:	File with data samples
		"""
		df = pd.read_csv(data_file)
		counts = np.asarray([len(df[df['label']==c]) for c in range(self.num_classes)])
		return np.argmax(counts).astype(np.int32)

	def batch_generator(self, file, shuffle=True):
		"""
		Generates batches for train operation
		Arguments:
		data_file:	pandas DataFrame
		shuffle:	Whether to shuffle input
		"""
		indices = np.arange(len(file))
		if shuffle:
			np.random.shuffle(indices)
		for i in range(np.int32(np.ceil(np.float(len(file))/self.batch_size))):
			batch = {}
			ulim = min((i+1)*self.batch_size, len(file))
			batch[0] = file.loc[indices[i*self.batch_size:ulim],file.columns!='label'].as_matrix()
			labels = file.loc[indices[i*self.batch_size:ulim],['label']].as_matrix().astype(np.int32)
			# np.concatenate(...) ensures number of columns == num_classes
			one_hot = pd.get_dummies(np.concatenate((labels.reshape((-1)),np.arange(self.num_classes))))
			batch[1] = one_hot.as_matrix()[:-self.num_classes].astype(np.float32)
			yield batch

	def split_dataset(self, data_file, preds, child_id):
		"""
		Split dataset for child nodes
		Arguments:
		data_file:	File containing the data in csv format. NOTE: pass original data only
		preds:		Decision maker predictions for each sample
		child_id:	List of child nodes (used in filename of split data)
		"""
		self.score = 0.0
		logging.debug('Split data file {}'.format(data_file))
		file = pd.read_csv(data_file)
		base = os.path.split(data_file)
		pred_class = np.argmax(preds, axis=1)
		if len(file)!=len(pred_class):
			logging.debug("len(file):{} len(pred_class):{}".format(len(file),len(pred_class)))
			logging.error("split_dataset : Array size mismatch")
			raise AssertionError("split_dataset : Array size mismatch")
		for j in range(self.output_dim):
			df = file.loc[pred_class==j]
			df_score = 0.0
			for cl in np.unique(df['label']):
				p = float(len(df.loc[df['label']==cl]))/len(df)
				df_score -= p * np.log2(p)
			self.score += float(len(df))/len(file)*df_score
			df.to_csv(os.path.join(base[0],'data_'+str(child_id[j])+'.csv'),index=False)
		logging.debug('Node impurity = {}'.format(self.score))

	def get_impurity(self):
		return self.score

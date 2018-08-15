import tensorflow as tf
import os
import numpy as np
import pandas as pd
import logging
from sklearn.cluster import KMeans
from classifiers.classifier import Classifier
from collections import Counter

class Perceptron(Classifier):
	"""Implement a single layer Perceptron"""

	def __init__(self, input_dim, output_dim, num_classes, decision_criterion, epochs, batch_size, node_id, data_balance):
		"""
		Arguments:
		input_dim: Dimension of input data
		output_dim: Dimension of output labels (equal to number of child nodes)
		num_classes: Number of classes in data
		decision_criterion: Decision function (CURRENTlY IMPLEMENTED: entropy, gini)
		epochs: Number of training epochs
		batch_size: Samples per batch (training as well as prediction)
		node_id: ID of node housing the decision maker
		data_balance: Ignored [balancing if any is done in the node]		
		"""
		super(Perceptron, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_classes = num_classes
		self.decision_criterion = decision_criterion
		self.epochs = epochs
		self.batch_size = batch_size
		self.node_id = node_id
		self.built = False
		self.graph = None
		self.impurity = None
		self.score = None

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
				self.W = tf.get_variable('weight', shape=[self.input_dim, self.output_dim], dtype=tf.float32,
					initializer=tf.zeros_initializer())
				self.b = tf.get_variable('bias', shape=[1, self.output_dim], dtype=tf.float32,
					initializer=tf.zeros_initializer())
				# q.shape = [None, output_dim]
				self.q = tf.nn.softmax(tf.matmul(self.data, self.W, name='matmul') + self.b, name='softmax')
			with tf.variable_scope('loss') as scope:
				# n.shape = [num_classes, output_dim]
				n = tf.matmul(tf.matrix_transpose(self.label, name='label_transpose'), self.q, name='n')
				# q.shape = [1, output_dim]
				q_sum = tf.reduce_sum(self.q, axis=0, keep_dims=True, name='q_sum')
				N = tf.reduce_sum(q_sum, name='N')
				# p.shape = [num_classes, output_dim]
				p = tf.divide(n, q_sum, name='p')
				clip_val_min = tf.Variable(1e-30, dtype=tf.float32, trainable=False)
				clip_val_max = tf.Variable(1, dtype=tf.float32, trainable=False)
				p_clip = tf.clip_by_value(p, clip_value_min=clip_val_min, clip_value_max=clip_val_max)
				# H.shape = [1, output_dim]
				H = -1 * tf.reduce_sum(tf.multiply(p_clip, tf.log(p_clip, name='log_p')), axis=0, keep_dims=True, name='Hj')
				self.loss = tf.reduce_sum(tf.multiply(H, q_sum/N), name='weighted_entropy')
			with tf.variable_scope('opt') as scope:
				# TODO: pass learning rate
				self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=global_step)

		self.built = True

	def train(self, data_file, balanced_file, child_id, data_path):
		"""
		Train on data and return params
		Arguments:
		data_file: File containing the original data
		balanced_file: File containing the balanced data
		child_id: List of child nodes (used for saving split data)
		data_path: Working directory where intermediate data is stored 
		"""
		if not self.built:
			logging.error("Perceptron: train called before build")
			raise AssertionError("Perceptron: train called before build")
		params = {}
		df = pd.read_csv(balanced_file)
		
		def _create_assign_op(self, data):
			kmeans_obj = KMeans(n_clusters=self.output_dim, n_init=1, n_jobs=-1)
			kmeans = kmeans_obj.fit(data.as_matrix([col for col in data.columns if col!='label']))
			assign_op = self.W.assign(kmeans.cluster_centers_.T)
			return assign_op

		assign_op = _create_assign_op(self, df)
		lr = 0.01 #np.float32(self.batch_size)/len(df)
		all_ok = False
		while not all_ok:
			with tf.Session(graph = self.graph) as sess:
				sess.run(tf.global_variables_initializer())
				sess.run(assign_op)
				prev_epoch_loss = None
				max_loss_drop = None
				initial_loss_drop = None
				patience = 0
				all_ok = True
				for e in range(self.epochs):
					epoch_loss = 0.0
					num_samples = 0
					for batch in self.batch_generator(df):
						_, bloss = sess.run([self.train_op, self.loss], feed_dict={self.data: batch[0], self.label:batch[1], self.lr:lr})
						epoch_loss += bloss*batch[0].shape[0]
						num_samples += batch[0].shape[0]
					if np.isnan(epoch_loss):
						logging.info("NaN loss. Restarting training.")
						lr *= 0.2
						all_ok = False
						break
					if e%20==0:
						logging.info("End of epoch {}".format(e+1))
						logging.info("Average epoch loss : {}".format(epoch_loss/num_samples))
					epoch_loss /= num_samples
					if e>=20 and e%10==0:
						if prev_epoch_loss is None:
							prev_epoch_loss = epoch_loss
						elif initial_loss_drop is None:
							initial_loss_drop = prev_epoch_loss - epoch_loss
							max_loss_drop = initial_loss_drop
						else:
							curr_loss_drop = prev_epoch_loss - epoch_loss
							prev_epoch_loss = epoch_loss
							if curr_loss_drop > max_loss_drop:
								patience = 0
								max_loss_drop = curr_loss_drop
								continue
							if curr_loss_drop < 0 or curr_loss_drop < 0.05*max_loss_drop:
								patience += 1
							if patience>=2:
								logging.info("Stopping training after {} epochs due to saturation of perceptron".format(e+1))
								break
				if not all_ok:
					continue
				logging.debug('Running predictions on {} for generating split'.format(data_file))
				preds = []
				
				dfo = pd.read_csv(data_file)
				for batch in self.batch_generator(dfo, shuffle=False):
					pred = sess.run(self.q, feed_dict={self.data: batch[0], self.label:batch[1]})
					preds += pred.tolist()
				self.split_dataset(data_file, np.asarray(preds), child_id, data_path)

				params['W'] = self.W.eval()
				params['b'] = self.b.eval()

		return params

	def predict(self, node_id, params, df, child_id):
		"""
		Predicts on dataframe
		Arguments:
		node_id: ID of node containing the decision_maker
		params:	Dictionary of decision_maker parameters
		df: DataFrame of test samples.
				NOTE: label column will be ignored. Assumes the indexing o dataframe
					is done using the assigned node i.e. samples reaching current node
					can be accessed by df.ix[self.node_id]
		child_id: List of child node IDs used to update the index
		"""
		if len(df)==0:
			return
		x = df.loc[node_id, (df.columns!='predicted_label') & (df.columns!='label') & (df.columns!='label_depth')].as_matrix()
		Wx = x.dot(params['W'])
		Wxb = Wx + params['b']
		preds = np.argmax(Wxb, 1)
		output = np.asarray(child_id)[preds.astype(np.int32)].tolist()

		as_list = np.asarray(df.index.tolist())
		idx = np.where(as_list==node_id)[0]
		as_list[idx] = output
		df.index = as_list

	def is_label(self, data_file, count_threshold, purity_threshold, data_path):
		"""
		Checks if the data should be split or declared a leaf node
		Arguments:
		data_file: File with data samples
		count_threshold: Minimum samples needed to consider tree growth
		purity_threshold: Percentage of most common class for purity-based stoppping of tree growth
		data_path: Working directory where intermediate data is stored
		"""
		df = pd.read_csv(data_file)
		if len(df) < count_threshold:
			logging.debug('Decide label node based on count_threshold')
			return True, 'count_threshold'
		# counts = np.asarray([len(df[df['label']==c]) for c in range(self.num_classes)]).astype(np.float32)
		counts = Counter(df['label'])
		class_prob = np.array(counts.values()).astype(np.float32)/len(df)
		self.impurity = -np.sum(class_prob*np.log2(class_prob))
		if np.float(counts.most_common(1)[0][1])/len(df) > purity_threshold:
			if len(counts) <= 1:
				# Only class in data "purity_threshold I"
				logging.debug('Decide label node based on purity')
				return True, 'purity_threshold I'
			# TODO: Maybe remove
			elif counts.most_common(2)[1][1] < len(df):
				# Majority class in data "purity_threshold II"
				logging.debug('Decide label node based on purity')
				return True, 'purity_threshold II'
		return False, ''

	def max_freq(self, data_file, data_path):
		"""
		Get most frequent class
		Arguments:
		data_file: File with data samples
		data_path: Working directory where intermediate data is stored
		"""
		df = pd.read_csv(data_file)
		counts = np.asarray([len(df[df['label']==c]) for c in range(self.num_classes)])
		return np.argmax(counts).astype(np.int32)

	def batch_generator(self, file, shuffle=True):
		"""
		Generates batches for train operation
		Arguments:
		file: pandas DataFrame
		shuffle: Whether to shuffle input
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

	def split_dataset(self, data_file, preds, child_id, data_path):
		"""
		Split dataset for child nodes
		Arguments:
		data_file: File containing the data in csv format. NOTE: pass original data only
		preds: Decision maker predictions for each sample
		child_id: List of child nodes (used in filename of split data)
		data_path: Working directory where intermediate data is stored
		"""
		self.score = 0.0
		logging.debug('Split data file {}'.format(data_file))

		file = pd.read_csv(data_file)
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
			df.to_csv(os.path.join(data_path, 'data', 'data_'+str(child_id[j])+'.csv'),index=False)
		logging.debug('Node impurity = {}'.format(self.score))

	def get_self_impurity(self):
		return self.impurity

	def get_split_impurity(self):
		return self.score
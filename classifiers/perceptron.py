import tensorflow as tf
import os
import numpy as np
import pandas as pd
from classifier import Classifier

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

	def build(self):
		"""
		Build classifier graph
		"""
		self.graph = tf.Graph()
		with self.graph.as_default():
			with tf.variable_scope('input') as scope:
				data = tf.placeholder(tf.float32, [None, self.input_dim], name="Xin")
				# one hot encoded sample vs target class
				label = tf.placeholder(tf.float32, [None, self.num_classes], name="Y")
				global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
			with tf.variable_scope('perceptron') as scope:
				W = tf.get_variable('weight', shape=[self.input_dim, self.output_dim], dtype=tf.float32,
					initializer=tf.contrib.layers.xavier_initializer())
				b = tf.get_variable('bias', shape=[1, self.output_dim], dtype=tf.float32,
					initializer=tf.contrib.layers.xavier_initializer())
				# q.shape = [None, output_dim]
				q = tf.nn.softmax(tf.matmul(data, W, name='matmul') + b, name='softmax')
			with tf.variable_scope('loss') as scope:
				# n.shape = [num_classes, output_dim]
				n = tf.matmul(tf.matrix_transpose(label, name='label_transpose'), q, name='n')
				# q.shape = [1, output_dim]
				q_sum = tf.reduce_sum(q, axis=0, keep_dims=True, name='q_sum')
				N = tf.reduce_sum(q_sum, name='N')
				# p.shape = [num_classes, output_dim]
				p = tf.divide(n, q_sum, name='p')
				# H.shape = [1, output_dim]
				H = -1 * tf.reduce_sum(tf.multiply(p, tf.log(p, name='log_p')), axis=0, keep_dims=True, name='Hj')
				loss = tf.reduce_sum(tf.multiply(H, q_sum/N), name='weighted_entropy')
			with tf.variable_scope('opt') as scope:
				# TODO: pass learning rate
				train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

		self.built = True

	def train(self, data_file, balanced_file, child_id):
		"""
		Train on data and return params
		Arguments:
		data_file:	File containing the original data
		balanced_file:	File containing the balanced data
		child_id:	List of child nodes (used for saving split data)
		"""
		assert self.built, "Perceptron: train called before build"
		params = {}
		self.session = tf.Session(graph = self.graph)
		with self.session as sess:
			sess.run(tf.global_variables_initializer())
			for e in range(self.epochs):
				epoch_loss = 0.0
				num_samples = 0
				for batch in self.batch_generator(balanced_file):
					_, bloss = sess.run([train_op, loss], feed_dict={data: batch[0], label:batch[1]})
					epoch_loss += bloss*batch[0].shape[0]
					num_samples += batch[0].shape[0]
				print "End of epoch {}".format(e+1)
				print "Average epoch loss : {}".format(epoch_loss/num_samples)

			preds = []
			for batch in self.batch_generator(data_file):
				pred = sess.run(q, feed_dict={data: batch[0], label:batch[1]})
				preds += pred.tolist()
			self.split_dataset(data_file, np.asarray(preds), child_id)

			params['W'] = W.eval()
			params['b'] = b.eval()

		return params

	def predict(self, node_id, params, data, child_id):
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
		x = df.ix[node_id, df.columns!='assigned_node' and df.columns!='label'].as_matrix()
		Wx = x * params['W']
		Wxb = Wx + b
		preds = np.argmax(Wxb, 1)
		output = np.asarray(child_id)[preds.astype(np.int32)].tolist()

		as_list = np.asarray(df.index.tolist())
		idx = np.where(as_list==node_id)[0]
		as_list[idx] = output
		df.index = as_list

	def is_label(self, data_file, count_threshold, purity_threshold):
		"""
		Decide if current node impurity is below threshold
		"""
		df = pd.read_csv(data_file)
		if len(df) < count_threshold:
			counts = np.asarray([len(df[df['label']==c]) for c in range(self.num_classes)]).astype(np.float32)
			if np.max(counts)/len(df) > purity_threshold:
				return True
		else:
			return False

	def max_freq(self, data_file):
		"""
		Decide if current node impurity is below threshold
		"""
		df = pd.read_csv(data_file)
		counts = np.asarray([len(df[df['label']==c]) for c in range(self.num_classes)])
		return np.argmax(counts).astype(np.int32)

	def batch_generator(self, data_file):
		"""
		Generates batches for train operation
		Arguments:
		data_file:	File containing the data in csv format
		"""
		file = pd.read_csv(data_file)
		indices = np.arange(len(file))
		np.random.shuffle(indices)
		for i in range(len(file)/self.batch_size):
			batch = {}
			ulim = min((i+1)*self.batch_size, len(file))
			batch[0] = file.loc[indices[i*self.batch_size:ulim]].as_matrix()[:,:-1]
			labels = file.loc[indices[i*self.batch_size:ulim]].as_matrix()[:,-1].astype(np.int32)
			# np.concatenate(...) ensures number of columns == num_classes
			one_hot = pd.get_dummies(np.concatenate((labels,np.arange(self.num_classes))))
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
		file = pd.read_csv(data_file)
		base = os.path.split(data_file)
		pred_class = np.argmax(preds, axis=1)
		assert len(file)==len(pred_class), "Perceptron : split_dataset : array size mismatch"
		for j in range(self.output_dim):
			indices = np.arange(len(file))[preds==j]
			df = file.loc[indices]
			df.to_csv(os.path.join(base[0],'data_'+str(child_id[j])+'.csv'),index=False)
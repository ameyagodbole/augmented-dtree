import tensorflow as tf
import os
import numpy as np
import pandas as pd
import logging
from classifiers.classifier import Classifier
from dataBalancing import DataBalance

import time
import pickle
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.cluster import KMeans
import math

class Perceptron_hybrid(Classifier):
	"""Implement a single layer Perceptron"""

	def __init__(self, input_dim, output_dim, num_classes, epochs, batch_size, node_id, data_balance):
		"""
		Arguments:
		input_dim:	Dimension of input data
		output_dim:	Dimension of output labels (equal to number of child nodes)
		num_classes: Number of classes in data
		epochs:		Number of training epochs
		batch_size:	Samples per batch (training as well as prediction)
		"""
		super(Perceptron_hybrid, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_classes = num_classes
		self.epochs = epochs
		self.batch_size = batch_size
		self.node_id = node_id
		self.built = False
		self.graph = None
		self.score = 0.0
		self.data_balance = data_balance
		self.features = None
		self.centers = []
		self.variance = []
		self.class_num = []
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
					initializer=tf.contrib.layers.xavier_initializer())
				self.b = tf.get_variable('bias', shape=[1, self.output_dim], dtype=tf.float32,
					initializer=tf.ones_initializer())
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
				clip_val_min = tf.Variable(1e-37, dtype=tf.float32, trainable=False)
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
		data_file:	File containing the original data
		balanced_file:	File containing the balanced data
		child_id:	List of child nodes (used for saving split data)
		"""
		if not self.built:
			logging.error("Perceptron: train called before build")
			raise AssertionError("Perceptron: train called before build")
		params = {}
		minority = []
		data_loc = os.path.join(data_path, 'perceptron_tree_hybrid','data',data_file)
		base = os.path.split(data_loc)[0]
		if self.data_balance:
			if self.node_id!=0:
				if not os.path.isfile(os.path.join(base,'data_{}.csv'.format(self.node_id))):
					logging.debug('No file to balance (data_{}.csv)'.format(self.node_id))
				else:
					logging.debug('Balance file data_{}.csv'.format(self.node_id))
					db = DataBalance(os.path.join(base,'data_{}.csv'.format(self.node_id)) , self.num_classes)
					minority = db.data_balance_hybrid(os.path.join(base,'b_data_{}.csv'.format(self.node_id)))
			else:
				if not os.path.isfile(os.path.join(data_path,data_file)):
					logging.debug('No file to balance (data_{}.csv)'.format(self.node_id))
				else:
					logging.debug('Balance file data_{}.csv'.format(self.node_id))
					db = DataBalance(os.path.join(data_path,data_file) , self.num_classes)
					minority = db.data_balance_hybrid(os.path.join(base,'b_data_{}.csv'.format(self.node_id)))
		else:
			if self.node_id!=0:
				db = DataBalance(os.path.join(base,'data_{}.csv'.format(self.node_id)) , self.num_classes)
				db.load()
			else:
				db = DataBalance(os.path.join(data_path,data_file) , self.num_classes)
				db.load()

			if db.data.empty:
				return
			minority = db.cluster_hybrid()
		
		if self.node_id==0:
			balance_file = os.path.join(data_path,data_file) 
			data_original = os.path.join(data_path,data_file) 
		else:
			balance_file = os.path.join(base,'b_data_{}.csv'.format(self.node_id)) if self.data_balance else os.path.join(base,'data_{}.csv'.format(self.node_id))
			data_original = os.path.join(base,'data_{}.csv'.format(self.node_id))
		df = pd.read_csv( balance_file)
		df_org = pd.read_csv(data_original)
		self.features = [col for col in df_org.columns if col!='label']
		# if df_org.shape[0]>6:
		# 		nn = 5
		# 	elif df_org[0]==1:
		# 		return None
		# 	else:
		# 		nn = df_org.shape[0]-1
		# 	nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(df_org[self.features])
		try:
			
			for i in minority:
				kmeans = None
				data = df_org.loc[df_org['label'] == i, self.features].as_matrix()
				if len(data)>4:
					kmeans = KMeans(n_clusters=int(0.3*len(data)), random_state=0).fit(data)
				elif len(data)>1:
					kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
				if len(data)>1:
					centers =  kmeans.cluster_centers_
					var = kmeans.inertia_

					for j in range(len(centers)):
						df_other_class = df_org.loc[df_org['label'] != i, self.features].as_matrix()
						dist = kmeans.transform(df_other_class)
						dist = dist[dist[:,j] < math.sqrt(var)]
						if len(dist)<0.3*len(data):
							self.centers.append(centers[j])
							self.variance.append(var)
							self.class_num.append(i)
		except TypeError:
			pass
		data_loc = os.path.join(data_path, 'perceptron_tree_hybrid','data','{}_node_{}_'.format(self.data_balance, self.node_id) )
		with open(data_loc + 'centers.pkl' , 'wb') as f:
			pickle.dump(self.centers, f)

		with open(data_loc + 'variance.pkl', 'wb') as f:
			pickle.dump(self.variance, f)

		with open(data_loc + 'class_num.pkl', 'wb') as f:
			pickle.dump(self.class_num, f)

		for i in range(self.num_classes):
			if i in minority:
				#print('df lenght: {}'.format(len(df)))
				#print('number of rows deleted: {}'.format(len(df.loc[df['label'] == i])))
				df_drop = df[ df['label'] ==i ]
				df = df.drop(df_drop.index, axis=0)
				#print('df lenght: {}'.format(len(df)))
		if self.node_id!=0:
			df.to_csv(balance_file,index=False)
			
		if self.node_id!=0:

			df = pd.read_csv(data_original)
			for i in range(self.num_classes):
				if i in minority:
					#print('df lenght: {}'.format(len(df)))
					#print('number of rows deleted: {}'.format(len(df.loc[df['label'] == i])))
					df_drop = df[ df['label'] ==i ]
					df = df.drop(df_drop.index, axis=0)
					#print('df lenght: {}'.format(len(df)))
			if len(df)>0:
				df.to_csv(data_original,index=False)

		df = pd.read_csv(balance_file)

		if len(df)==0:
			df = pd.read_csv(data_original)

		lr = np.float32(self.batch_size)/len(df)
		all_ok = False


		while not all_ok:
			with tf.Session(graph = self.graph) as sess:
				sess.run(tf.global_variables_initializer())
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
					if e%10==0:
						logging.info("End of epoch {}".format(e+1))
						logging.info("Average epoch loss : {}".format(epoch_loss/num_samples))
					epoch_loss /= num_samples
					if e>=10 and e%10==0:
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
							if curr_loss_drop < 0 or curr_loss_drop < 0.1*max_loss_drop:
								patience += 1
							if patience>=2:
								logging.info("Stopping training after {} epochs due to saturation of perceptron".format(e+1))
								break
				if not all_ok:
					continue
				logging.debug('Running predictions on {} for generating split'.format(data_file))
				preds = []
				if self.node_id == 0:
					data_loc = os.path.join(data_path, data_file)
				else:
					data_loc = os.path.join(data_path,'perceptron_tree_hybrid','data',data_file)
				dfo = pd.read_csv(data_loc)
				for batch in self.batch_generator(dfo, shuffle=False):
					pred = sess.run(self.q, feed_dict={self.data: batch[0], self.label:batch[1]})
					preds += pred.tolist()
				self.split_dataset(data_file, np.asarray(preds), child_id, data_path)

				params['W'] = self.W.eval()
				params['b'] = self.b.eval()

		return params

	def predict(self, node_id, params, df, child_id, data_path):
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
		if len(df)==0:
			return
		features = [col for col in df.columns if col not in ['label','assigned_node','predicted_label']]
		x = df.ix[node_id, features].as_matrix()
		Wx = x.dot(params['W'])
		Wxb = Wx + params['b']
		preds = np.argmax(Wxb, 1)
		output = np.asarray(child_id)[preds.astype(np.int32)].tolist()

		as_list = np.asarray(df.index.tolist())
		idx = np.where(as_list==node_id)[0]
		as_list[idx] = output
		df.index = as_list
		data_loc = os.path.join(data_path, 'perceptron_tree_hybrid','data','{}_node_{}_'.format(self.data_balance, self.node_id) )
		logging.debug('loading extreme minority model')
		try:
			with open (data_loc+ 'centers.pkl', 'rb') as fp:
				self.centers = pickle.load(fp)
			with open (data_loc+ 'variance.pkl', 'rb') as fp:
				self.variance = pickle.load(fp)
			with open (data_loc+ 'class_num.pkl', 'rb') as fp:
				self.class_num = pickle.load(fp)

			print(len(self.centers))
			X = df.loc[node_id, (df.columns!='predicted_label') & (df.columns!='label')]
			d = cdist(X.toarray(),centers)
			l = np.where(d<math.sqrt(self.variance[0])/2)
			logging.debug(len(l))
			for loc in l:
				df.loc[self.node_id,'predicted_label'][loc[1]] = class_num[loc[0]]
				logging.debug(class_num[loc[0]])
				df.loc[self.node_id][loc[0]].index = self.node_id
		except IOError:
			pass

	def is_label(self, data_file, count_threshold, purity_threshold, data_path):
		"""
		Checks if the data should be split or declared a leaf node
		Arguments:
		data_file:	File with data samples
		count_threshold:	Minimum samples needed to consider tree growth
		purity_threshold:	Percentage of most common class for purity-based stoppping of tree growth
		"""
		if self.node_id == 0:
			data_loc = os.path.join(data_path, data_file)
		else:
			data_loc = os.path.join(data_path,'perceptron_tree_hybrid','data',data_file)

		df = pd.read_csv(data_loc)
		if len(df) < count_threshold:
			logging.debug('Decide label node based on count_threshold')
			return True
		counts = np.asarray([len(df[df['label']==c]) for c in range(self.num_classes)]).astype(np.float32)
		if np.float(np.max(counts))/len(df) > purity_threshold:
			logging.debug('Decide label node based on purity')
			return True
		return False

	def max_freq(self, data_file, data_path):
		"""
		Get most frequent class
		Arguments:
		data_file:	File with data samples
		"""
		if self.node_id == 0:
			data_loc = os.path.join(data_path, data_file)
		else:
			data_loc = os.path.join(data_path,'perceptron_tree_hybrid','data',data_file)

		df = pd.read_csv(data_loc)
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

	def split_dataset(self, data_file, preds, child_id, data_path):
		"""
		Split dataset for child nodes
		Arguments:
		data_file:	File containing the data in csv format. NOTE: pass original data only
		preds:		Decision maker predictions for each sample
		child_id:	List of child nodes (used in filename of split data)
		"""
		self.score = 0.0
		logging.debug('Split data file {}'.format(data_file))

		if self.node_id == 0:
			data_loc = os.path.join(data_path, data_file)
		else:
			data_loc = os.path.join(data_path,'perceptron_tree_hybrid','data',data_file)
		#print(data_loc)
		file = pd.read_csv(data_loc)
		base = os.path.split(data_file)
		#print(base[0])
		pred_class = np.argmax(preds, axis=1)
		#print('file length:{}'.format(len(file)))
		#print('len of pred class: {}'.format(len(pred_class)))
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
			#print('saving')
			#print(os.path.join(data_path, 'perceptron_tree','data','data_'+str(child_id[j])+'.csv'))
			df.to_csv(os.path.join(data_path, 'perceptron_tree_hybrid','data','data_'+str(child_id[j])+'.csv'),index=False)
		logging.debug('Node impurity = {}'.format(self.score))

	def get_impurity(self):
		return self.score

	def get_self_impurity(self, data_path, data_file):		
		if self.node_id == 0:	
			data_loc = os.path.join(data_path, data_file)
		else:
			data_loc = os.path.join(data_path,'perceptron_tree_hybrid','data',data_file)
		df = pd.read_csv(data_loc)
		impurity_score = 0.0		
		for cl in np.unique(df['label']):
			p = float(len(df.loc[df['label']==cl]))/len(df)	
			impurity_score = impurity_score - p * np.log2(p)
		return impurity_score

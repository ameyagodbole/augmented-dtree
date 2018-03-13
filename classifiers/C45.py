import os
import numpy as np
import pandas as pd
import logging
import csv
from csv import reader
from classifiers.classifier import Classifier

class C45(Classifier):
	"""Implement a C4.5 classifier"""

	def __init__(self, input_dim, output_dim, num_classes, epochs, batch_size):
		"""
		Arguments:
		input_dim:	Dimension of input data
		output_dim:	Dimension of output labels (equal to number of child nodes)
		num_classes: Number of classes in data		
		epochs:		Ignored
		batch_size:	Ignored
		"""
		super(C45, self).__init__()
		self.input_dim = input_dim
		self.output_dim = 2
		self.num_classes = num_classes
		self.index = None
		self.value = None
		self.score = None
		self.impurity_drop = None
		self.b_score = None

		
	def build(self):
		pass

	# Calculate the impurity index for a split dataset
	def impurity_index(self, groups, classes):
		# count all samples at split point
		n_instances = float(sum([len(group) for group in groups]))
		# sum weighted impurity index for each group
		impurity = 0.0
		for group in groups:
			size = float(len(group))
			# avoid divide by zero
			if size == 0:
				continue
			score = 0.0
			# score the group based on the score for each class
			for class_val in classes:
				p = np.float(list(group['label']).count(class_val)) / size
				if p != 0.0:
					score -= p * np.log2(p)
			# weight the group score by its relative size
			impurity += score * (size / n_instances)
		return impurity

	def eval_label_col(self, dataset, class_values, index):
		a = dataset[index].unique()
		if index!='label':	
			print(index)			
			[self.eval_label_row(dataset, class_values, index, x) for x in a]
		

	def eval_label_row(self, dataset, class_values, index, value):
		
		groups = (dataset.loc[dataset[index]<value],dataset.loc[dataset[index]>=value])
		impurity = self.impurity_index(groups, class_values)
		if impurity < self.b_score:
			self.b_score = impurity
			self.index = index
			self.value = value



	def get_split(self, balanced_file):
		"""
		Decides best split to minimize entropy. 'Value' is threshold to split along 'index'
		"""	
		dataset = pd.read_csv(balanced_file)
		#dataset = dataset.head(n = 50)
		class_values = np.unique(dataset['label'])
		self.index, self.value, self.b_score = 999, 999, float('inf')
		[self.eval_label_col(dataset, class_values, index) for index in dataset.columns]
		print(self.value)
		print(self.index)
		


	def split_dataset(self, data_file, child_id):
		"""
		Split dataset for child nodes
		Arguments:
		data_file:	File containing the data in csv format. NOTE: pass original data only
		preds:		Decision maker predictions for each sample
		child_id:	List of child nodes (used in filename of split data)
		"""
		base = os.path.split(data_file)
		dataset = pd.read_csv(data_file)
		if np.sum([dataset[self.index]<self.val])==0:
			group1 = pd.DataFrame()  
		else:
			group1 = dataset.loc[dataset[self.index]<self.value]

		if np.sum([dataset[self.index]>=self.value])==0:
			group2 = pd.DataFrame()  
		else:
			group2 =dataset.loc[dataset[self.index]>=self.value]

		groups = (group1,group2)
		class_values = np.unique(dataset['label'])
		self.score = self.impurity_index(groups, class_values)
		
		for j in range(self.output_dim):
			groups[j].to_csv(os.path.join(base[0],'data_'+str(child_id[j])+'.csv'),index=False)

			

	def train(self, data_file, balanced_file, child_id):		
		self.get_split(balanced_file)
		self.split_dataset(data_file, child_id)
		params = {}
		params['index'] = self.index
		params['value'] = self.value
		return params

	def is_label(self, data_file, count_threshold, purity_threshold):
		"""
		Checks if the data should be split or declared a leaf node
		Arguments:
		data_file:	File with data samples
		count_threshold:	Maximum samples needed to consider purity-based stoppping of tree growth
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

	def get_impurity(self):
		return self.score

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
		print(node_id)
		if node_id==44:
			return
		cols = [col for col in df.columns if col not in ['predicted_label', 'label','assigned_node']]
		x = df.ix[node_id, cols]
		
		preds = []
		for index, row in x.iterrows():
			if row[params['index']] < params['value']:
				preds.append(0)
			else:
				preds.append(1)
		
		
		output = np.asarray(child_id)[preds].tolist()

		as_list = np.asarray(df.index.tolist())
		idx = np.where(as_list==node_id)[0]
		as_list[idx] = output
		df.index = as_list

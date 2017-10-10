import os
import numpy as np
import pandas as pd
import csv
from csv import reader
from classifiers.classifier import Classifier

class C45(Classifier):
	"""Implement a C4.5 classifier"""

	def __init__(self, input_dim, output_dim, num_classes, epochs, batch_size, t1, t2):
		"""
		Arguments:
		input_dim:	Dimension of input data
		output_dim:	Dimension of output labels (equal to number of child nodes)
		num_classes: Number of classes in data		
		t1:		Threshold for impurity score
		t2:		Threshold for size
		"""
		super(C45, self).__init__()
		self.input_dim = input_dim
		self.output_dim = 2
		self.num_classes = num_classes
		self.groups = null
		self.index = null
		self.split_val = null
		self.score = null
		self.impurity_drop
		self.t1 = t1
		self.t2 = t2

		

	def build(self):
		pass

	def train(self, data_file, balanced_file, child_id):
		dataset = pd.read_csv(balanced_file)
		get_split(dataset)
		split_dataset(data_file, child_id)
		params = {}
		params['index'] = self.index
		params['value'] = self.split_val
		return params

	def load_csv(filename):
	
	# Calculate the Gini index for a split dataset
	def gini_index(groups, classes):
		# count all samples at split point
		n_instances = float(sum([len(group) for group in groups]))
		# sum weighted Gini index for each group
		gini = 0.0
		for group in groups:
			size = float(len(group))
			# avoid divide by zero
			if size == 0:
				continue
			score = 0.0
			# score the group based on the score for each class
			for class_val in classes:

				p = [row[-1] for row in group].count(class_val) / size
				if p != 0.0:
					score -= p * np.log2(p)
			# weight the group score by its relative size
			gini += score * (size / n_instances)
		return gini

	def test_split(index, value):
		"""
		Forms two groups for a given value and index.
		"""
		left, right = list(), list()
		for row in self.dataset:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right

	def get_split(self, dataset):
		"""
		Decides best split to minimize entropy. 'Value' is threshold to split along 'index'
		"""
		class_values = list(set(row[-1] for row in dataset))
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		for index in range(len(dataset[0])-1):
			for row in dataset:
				groups = test_split(index, row[index])
				gini = gini_index(groups, class_values)
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups
		self.value = b_value
		self.index = b_index
		self.groups = b_groups
		self.score = b_score


	def split_dataset(self, data_file, child_id):
		"""
		Split dataset for child nodes
		Arguments:
		data_file:	File containing the data in csv format. NOTE: pass original data only
		preds:		Decision maker predictions for each sample
		child_id:	List of child nodes (used in filename of split data)
		"""
		base = os.path.split(data_file)
		
		for j in range(self.output_dim):
			with open(os.path.join(base[0],'data_'+str(child_id[j])+'.csv'), "wb") as f:

				writer = csv.writer(f)
				writer.writerows(self.groups[j])

	def is_label(self, data_file, count_threshold, purity_threshold):
		"""
		Checks if the data should be split or declared a leaf node
		"""
		dataset = pd.read_csv(data_file)

		classes = list(set(row[-1] for row in dataset))
		size = float(len(dataset))
		# avoid divide by zero
		if size == 0:
			return True
		
		p = []
		for class_val in classes:
			p.append([row[-1] for row in dataset].count(class_val) / size)
		r = np.argmax(p)
		q = 1 - p[r]

		if size < count_threshold and  q < purity_threshold:
			return True

		return False



	def max_freq(self, data_file):
		dataset = pd.read_csv(data_file)

		classes = list(set(row[-1] for row in dataset))
			
		freq = []
		for class_val in classes:
			freq.append([row[-1] for row in dataset].count(class_val))
		return np.argmax(freq)

	def get_impurity(self):
		return self.score

	def get_impurity_drop(self):
		pass

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
		for index, row in df.iterrows():   		
			if row[params['index']] < params['value']:
				preds.append(0)
			else:
				preds.append(1)
		
		output = np.asarray(child_id)[preds.astype(np.int32)].tolist()

		as_list = np.asarray(df.index.tolist())
		idx = np.where(as_list==node_id)[0]
		as_list[idx] = output
		df.index = as_list

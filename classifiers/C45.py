import os
import numpy as np
import pandas as pd
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
		"""
		super(C45, self).__init__()
		self.input_dim = input_dim
		self.output_dim = 2
		self.num_classes = num_classes
		self.dataset = null
		self.groups = null
		self.index = null
		self.split_val = null
		self.score = null
		

	def build(self):
		pass

	def train(self, data_file, balanced_file, child_id):
		load_csv(balanced_file)
		get_split()
		split_dataset(data_file, child_id)
		params = {}
		params['index'] = self.index
		params['value'] = self.split_val
		return params

	def load_csv(filename):
		file = open(filename, "rb")
		lines = reader(file)
		self.dataset = list(lines)
	
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

	def get_split():
		"""
		Decides best split to minimize entropy. 'Value' is threshold to split along 'index'
		"""
		class_values = list(set(row[-1] for row in self.dataset))
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		for index in range(len(self.dataset[0])-1):
			for row in self.dataset:
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

	def is_label(self):
		"""Checks if the data should be split or declared a leaf node"""
		pass
	def max_freq(self):
		pass



	def predict(params, data):
		"""
		Split dataset for child nodes
		Arguments:
		data_file:	File containing the data in csv format. NOTE: pass original data only
		preds:		Decision maker predictions for each sample
		"""
		base = os.path.split(data_file)
		f = open(param_file, "r")
		index = int(f.readline())
		value = int(f.readline())
		if data[index] < value:
			return(0)
		else:
			return(1)
			

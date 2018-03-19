import pandas as pd
import os
import pickle
import logging
from classifiers.classifier import Classifier

class DTNode(object):
	"""
	DTNode class to define DTree decisions
	"""
	
	def __init__(self, node_id, parent_id, node_depth, num_classes, num_child, working_dir_path = None,
		data_file=None, balanced_file=None, count_threshold=0, purity_threshold=1.):
		"""
		Arguments:
		node_id: Index of node in tree nodelist
		parent_id: Index of parent node in tree nodelist
		node_depth: Depth of node in decision tree
		num_classes: Number of classes in data
		num_child: Number of child nodes of node
		working_dir_path: Working directory used to store intermediate files and results
		data_file: Original data
		balanced_file: Balanced data
		count_threshold: Minimum number of samples needed, otherwise node is marked as leaf
		purity_threshold: Percentage of most common class for purity-based stoppping of tree growth
		"""
		super(DTNode, self).__init__()
		self.node_id = node_id
		self.parent_id = parent_id
		self.node_depth = node_depth
		self.is_decision_node = False
		self.label = None
		self.label_type = ''
		self.num_classes = num_classes
		self.num_child = num_child
		self.child_id = []
		self.data_file = data_file
		self.working_dir_path = working_dir_path
		self.balanced_file = balanced_file
		self.decision_maker = None
		self.count_threshold = count_threshold
		self.purity_threshold = purity_threshold

		self.params = {}
		self.trained = False
	
	def set_decision_maker(self, decision_maker):
		"""
		Set the decision maker for the current node
		Arguments:
		decision_maker:	Object of type Classifier
		"""
		if not isinstance(decision_maker, Classifier):
			raise TypeError("DTNode received decision_maker of type: {}".format(type(decision_maker)))
		self.decision_maker = decision_maker

	def set_child_id_start(self, child_id_start):
		"""
		Set the child_id_start for the current node
		Arguments:
		child_id_start:	Starting id of child nodes
		"""
		if child_id_start < 0:
			raise IndexError("DTNode received negative child_id_start : {}".format(child_id_start))
		self.child_id = [child_id_start+i for i in range(self.num_child)]

	def train(self):
		"""
		Train on data and save params in Node
		"""
		if self.is_label_node() or self.num_child==0 or self.num_child==1:
			if(self.num_child==0):
				logging.debug('Set {} to label based on depth'.format(self.node_id))
			if(self.num_child==1):
				logging.debug('Set {} to label based on having only one child node'.format(self.node_id))
			self.label = self.get_label()
			self.num_child = 0
			self.child_id = []
			return self.child_id
		self.decision_maker.build()
		self.params = self.decision_maker.train(self.data_file, self.balanced_file, self.child_id, self.working_dir_path)
		return self.child_id

	def save_node_params(self, savepath):
		"""
		Save node params to file
		Arguments:
		savepath:	Directory to save node params dictionary
		"""
		with open(os.path.join(savepath, 'node_{}.pkl'.format(self.node_id)), 'wb') as savefile:
			pickle.dump(self.params, savefile, protocol=pickle.HIGHEST_PROTOCOL)

	def load_node_params(self, path):
		"""
		Load node params to file
		Arguments:
		path:	Path to saved params file
		"""
		with open(os.path.join(path, 'node_{}.pkl'.format(self.node_id)), 'rb') as f:
			self.params = pickle.load(f)

	def predict(self, df):
		"""
		Predict on dataframe. If not a label node, returns relevant child node id
		Arguments:
		df: DataFrame of test samples.
				NOTE: label column will be ignored. Assumes the indexing o dataframe
					is done using the assigned node i.e. samples reaching current node
					can be accessed by df.ix[self.node_id]
				NOTE: decision will be placed in predicted_label column
		"""
		if self.is_decision_node or len(self.child_id)==0:
			#df.ix[self.node_id,'predicted_label'] = self.label 
			
			try:
 				# df.ix[self.node_id,'predicted_label']
 				df.loc[self.node_id,'predicted_label'].shape 
			except KeyError:
				logging.debug('no data for this node_id')
				return
			df.loc[self.node_id,'predicted_label'] = self.label	
			df.loc[self.node_id,'label_depth'] = self.node_depth + 1	
		else:
			try:
				# df.ix[self.node_id,'predicted_label']
				self.decision_maker.predict(self.node_id, self.params, df, self.child_id)
			except KeyError:
				logging.debug('no data for this node_id')
				return
			# self.decision_maker.predict(self.node_id, self.params, df, self.child_id)

	def is_label_node(self):
		"""
		Check if current node is label node
		"""
		self.is_decision_node, self.label_type = self.decision_maker.is_label(self.data_file, self.count_threshold, self.purity_threshold, self.working_dir_path) 
		return self.is_decision_node

	def get_label(self):
		"""
		Set label for decision node
		"""
		return self.decision_maker.max_freq(self.data_file, self.working_dir_path)
	
	def get_self_impurity(self):
		"""
		Calculate impurity of node
		"""
		return self.decision_maker.get_self_impurity()

	def get_split_impurity(self):
		"""
		Calculate impurity of split generated by trained node
		"""
		return self.decision_maker.get_split_impurity()

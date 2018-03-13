import pandas as pd
import os
import pickle
import logging
from classifiers.classifier import Classifier
from scipy.spatial.distance import cdist
from classifiers.perceptron import Perceptron
from classifiers.perceptron_hybrid import Perceptron_hybrid


class DTNode(object):
	"""
	DTNode class to define DTree decisions
	"""
	
	def __init__(self, node_id, parent_id, node_depth, num_classes, num_child, data_path = None,
		data_file=None, balanced_file=None, count_threshold=None, purity_threshold=None, data_balance = None):
		"""
		Arguments:
		node_id:	Index of node in tree nodelist
		parent_id:	Index of parent node in tree nodelist
		node_depth: Depth of node in decision tree
		num_classes:	Number of classes in data
		num_child:	Number of child nodes of node
		data_file:	Original data
		balanced_file:	Balanced data
		count_threshold:	Maximum samples needed to consider purity-based stoppping of tree growth
		purity_threshold:	Percentage of most common class for purity-based stoppping of tree growth
		"""
		super(DTNode, self).__init__()
		self.node_id = node_id
		self.parent_id = parent_id
		self.node_depth = node_depth
		self.is_decision_node = False
		self.label = None
		self.num_classes = num_classes
		self.num_child = num_child
		self.child_id = []
		self.data_file = data_file
		self.data_path = data_path
		self.balanced_file = balanced_file
		self.decision_maker = None
		self.count_threshold = count_threshold
		self.purity_threshold = purity_threshold
		self.data_balance = data_balance

		self.params = {}
		self.trained = False
		self.centers = []
		self.variance = []
		self.class_num= []
	
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
		if self.is_label_node() or self.num_child==0:
			if(self.num_child==0):
				logging.debug('Set {} to label based on depth'.format(self.node_id))
			if(self.num_child==1):
				logging.debug('Set {} to label based on having only one child node'.format(self.node_id))
			self.is_decision_node = True
			self.label = self.get_label()
			self.num_child = 0
			self.child_id = []
			return self.child_id
		self.decision_maker.build()
		self.params = self.decision_maker.train(self.data_file, self.balanced_file, self.child_id, self.data_path)
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

	def predict(self, df, data_path):
		"""
		Predict on dataframe. If not a label node, returns relevant child node id
		Arguments:
		df:		DataFrame of test samples.
				NOTE: label column will be ignored. Assumes the indexing o dataframe
					is done using the assigned node i.e. samples reaching current node
					can be accessed by df.ix[self.node_id]
				NOTE: decision will be placed in predicted_label column
		"""
		if self.is_decision_node or len(self.child_id)==0:
			#df.ix[self.node_id,'predicted_label'] = self.label 
		# 	for i , row in df.iterrows():
		# 		if i == self.node_id:
		# 			row['predicted_label'] = self.label
		# 			logging.debug('label{} assigned'.format(self.label))
			try:
				df.loc[self.node_id,'predicted_label'] = self.label
			except ValueError:
				return
		# # 	try:
 				
		# 		if self.decision_maker == Perceptron:

		# 			data_loc = os.path.join(data_path, 'perceptron_tree','data','node_{}_'.format(self.node_id) )

		# 		else:
		# 			data_loc = os.path.join(data_path, 'perceptron_tree_hybrid','data','{}_node_{}_'.format(self.data_balance, self.node_id) )
		# 		try:
		# 			with open (data_loc+ 'centers.pkl', 'rb') as fp:
		# 				self.centers = pickle.load(fp)
		# 			with open (data_loc+ 'variance.pkl', 'rb') as fp:
		# 				self.variance = pickle.load(fp)
		# 			with open (data_loc+ 'class_num.pkl', 'rb') as fp:
		# 				self.class_num = pickle.load(fp)
		# 			X = df.loc[self.node_id, (df.columns!='predicted_label') & (df.columns!='label')]

		# 			lable_list = []

		# 			for index, row in X.iterrows():
		# 				Y = pd.np.reshape(row.as_matrix(),(1,180))
						
		# 				dist_row = cdist(Y,self.centers)
		# 				idx = dist_row[0].tolist().index(min(dist_row[0]))
		# 				lable_list.append(self.class_num[idx])

		# 			df.ix[self.node_id,'predicted_label'] = lable_list	
		# 		except IOError:
		# 			pass
		# 	except ValueError:
		# 		logging.info("No data for this node")

		else:
			try:
				# df.ix[self.node_id,'predicted_label']
				self.decision_maker.predict(self.node_id, self.params, df, self.child_id, data_path)
			except KeyError:
				logging.debug('no data for this node_id: {}'.format(self.node_id))
				return
			# self.decision_maker.predict(self.node_id, self.params, df, self.child_id)

	def is_label_node(self):
		"""
		Check if current node is label node

		"""
		
		return self.decision_maker.is_label(self.data_file, self.count_threshold, self.purity_threshold, self.data_path) 

	def get_label(self):
		"""
		Set label for decision node
		"""
		return self.decision_maker.max_freq(self.data_file, self.data_path)
	
	def get_impurity(self):
		"""
		Set label for decision node
		"""
		return self.decision_maker.get_impurity()
	def get_impurity(self):
		
		"""
		
		Set label for decision node
		
		"""
		
		child_impurity = self.decision_maker.get_impurity()
		
		self_impurity = self.decision_maker.get_self_impurity(self.data_path, self.data_file)
		
		return self_impurity - child_impurity


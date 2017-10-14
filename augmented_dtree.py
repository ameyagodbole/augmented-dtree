import os
from augmented_dtree_node import DTNode
from classifiers.classifier import Classifier
from classifiers.perceptron import Perceptron
from classifiers.C45 import C45
from dataBalancing import DataBalance
from pkg_logger import *

class DTree(object):
	"""DTree class to store tree structure"""
	
	def __init__(self, num_classes, num_child, max_depth, data_type, data_dimension, data_balance, decision_type,
	 count_threshold, purity_threshold, impurity_drop_threshold, verbosity=2):
		"""
		Arguments:
		num_classes: Number of classes in data
		num_child:	Numer of child nodes per decision node
		max_depth:	Maximum depth of decision tree
		data_type:	One of {'numeric','image'}
		data_dimension:	Number of features in data; int for numeric , tuple of ints for image
		data_balance: Bool (whether to use data_balancing)
		decision_type: Classifier to be used
		verbosity:	0-ERROR 1-INFO 2-DEBUG
		"""
		super(DTree, self).__init__()
		self.num_classes = num_classes
		self.num_child = num_child
		self.max_depth = max_depth
		self.data_type = data_type
		self.data_dimension = data_dimension
		self.data_balance = data_balance
		self.decision_type = decision_type
		self.nodes = []
		self.built = False
		self.count_threshold = count_threshold
		self.purity_threshold = purity_threshold
		self.impurity_drop_threshold = impurity_drop_threshold

		if verbosity==2:
			logging.getLogger('').setLevel(logging.DEBUG)
		elif verbosity==1:
			logging.getLogger('').setLevel(logging.INFO)
		else:
			logging.getLogger('').setLevel(logging.ERROR)

		if self.decision_type == 'Perceptron':
			self.decision_type = Perceptron
		elif self.decision_type == 'C45':
			self.decision_type = C45
			logging.warning('num_child overide to 2 for C4.5')
			self.num_child = 2
		else:
			raise NotImplementedError('Feature not implemented')

	def train(self, data_file, epochs_per_node, batch_size, model_save_path):
		"""
		Build tree and save node parameters
		Arguments:
		data_file:	Data file for root node. Other nodes create their own file
		epochs_per_node: Number of training epochs for each node
		batch_size:	Batch size for training and predictions
		model_save_path:	Directory to save node parameters
		"""
		logging.debug('Train called')
		base = os.path.split(data_file)[0]
		node_to_process = 0
		new_node = DTNode(node_id=0, parent_id=0, node_depth=0, num_classes=self.num_classes,
		 num_child=self.num_child, data_file=data_file, balanced_file = data_file, 
		 count_threshold = self.count_threshold, purity_threshold = self.purity_threshold)
		self.nodes.append(new_node)
		while True:
			try:
				curr_node = self.nodes[node_to_process]
			except IndexError:
				logging.info("{} nodes processed. Tree building done.".format(len(self.nodes)))
				break
			logging.info('Process node {}'.format(node_to_process))
			curr_node.set_decision_maker(self.decision_type(input_dim=self.data_dimension, output_dim=self.num_child,
				 num_classes=self.num_classes, epochs=epochs_per_node, batch_size=batch_size))
			curr_node.set_child_id_start(len(self.nodes))
				
			child_list = curr_node.train()
			curr_node.save_node_params(model_save_path)

			if self.data_balance:
				if not os.path.isfile(os.path.join(base,'data_{}.csv'.format(node_to_process+1))):
					logging.debug('No file to balance (data_{}.csv)'.format(node_to_process+1))
				else:
					logging.debug('Balance file data_{}.csv'.format(node_to_process+1))
					db = DataBalance(os.path.join(base,'data_{}.csv'.format(node_to_process+1)) , self.num_classes)
					db.data_balance(os.path.join(base,'b_data_{}.csv'.format(node_to_process+1)))

			if child_list == []:
				logging.debug('No child nodes for node {}'.format(node_to_process))
				node_to_process += 1
				continue

			if self.get_impurity_drop(self.nodes[curr_node.parent_id], curr_node) < self.impurity_drop_threshold :
				curr_node.child_id = []
				logging.debug('Stop growth at node {} due to low impurity drop rate'.format(node_to_process))
				node_to_process += 1
				continue
			
			for i in child_list:
				# stop tree growth at max_depth
				num_child = self.num_child if self.max_depth>1+curr_node.node_depth else 0
				# if not using data balancing, send original file as balanced file
				balance_filename = 'b_data_{}.csv'.format(i) if self.data_balance else 'data_{}.csv'.format(i) 
				
				new_node = DTNode(node_id=i, parent_id=curr_node.node_id, node_depth=1+curr_node.node_depth,
				 num_classes=self.num_classes, num_child=num_child, data_file=os.path.join(base,'data_{}.csv'.format(i)),
				 balanced_file=os.path.join(base,balance_filename),
				 count_threshold = self.count_threshold, purity_threshold = self.purity_threshold)
				
				self.nodes.append(new_node)				
			
			node_to_process += 1
		self.built = True

	def save(self, model_save_file):
		"""
		Save tree structure. NOTE: Node parameters are saved separately
		Arguments:
		model_save_file:	File to save tree structure
		"""
		logging.debug('Save called')
		structure = {}
		for i in self.nodes:
			node_info = {}
			node_info['parent_id'] = i.parent_id
			node_info['num_child'] = i.num_child
			node_info['child_id'] = i.child_id
			node_info['node_depth'] = i.node_depth
			node_info['is_decision_node'] = i.is_decision_node
			node_info['label'] = i.label
			node_info['impurity'] = i.get_impurity()
			structure[i.node_id] = node_info
		logging.debug('Saving to {}'.format(model_save_file))
		with open(os.path.join(model_save_file), 'wb') as savefile:
			pickle.dump(structure, savefile, protocol=pickle.HIGHEST_PROTOCOL)

	def load_tree(self, model_save_file, model_save_path):
		"""
		Load tree from structure file.
		Arguments:
		model_save_file:	File to save tree structure
		model_save_path:	Directory to save node parameters
		"""
		logging.debug('Load called')
		structure = {}
		with open(os.path.join(model_file), 'rb') as modelfile:
			structure = pickle.load(f)

		self.nodes = [None for _ in range(len(structure.keys()))]
		for i in structure.keys():
			logging.info('Loading node {}'.format(i))
			self.nodes[i] = DTNode(node_id=i, parent_id=structure[i]['parent_id'], node_depth=structure[i]['node_depth'],
				 num_classes=self.num_classes, num_child=structure[i]['num_child'])
			curr_node = self.nodes[i]
			curr_node.set_decision_maker(self.decision_type(input_dim=self.data_dimension, output_dim=self.num_child,
				 num_classes=self.num_classes, epochs=epochs_per_node, batch_size=batch_size))
			curr_node.child_id = structure[i]['child_id']
			curr_node.is_decision_node = structure[i]['is_decision_node']
			curr_node.label = structure[i]['label']
			curr_node.load_node_params(model_save_path)

	def predict(self, model_save_file, model_save_path, data_file):
		"""
		Iteratively predict on test data.
		Arguments:
		model_save_file:	File with saved tree structure
		model_save_path:	Directory with saved node parameters
		data_file:		Data file of test samples.
						NOTE: label column will be ignored. Assumes the indexing o dataframe
							is done using the assigned node i.e. samples reaching current node
							can be accessed by df.ix[self.node_id]
						NOTE: decision will be placed in predicted_label column of data_file
		"""
		logging.debug('Predict called')
		self.load_tree(model_save_file, model_save_path)
		df = pd.read_csv(data_file, index_col='assigned_node')
		df['predicted_label'] = [0 for _ in range(len(df))]
		for node in self.nodes:
			node.predict(df)

	def get_impurity_drop(self, parent_node, child_node):
		"""
		Find impurity drop from parent to child node.
		Arguments:
		parent_node:	Parent node
		child_node:		Child node	
		"""
		if child_node.node_id == 0:
			return float('inf')
		return (parent_node.get_impurity() - child_node.get_impurity())

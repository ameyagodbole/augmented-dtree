import os
from augmented_dtree_node import DTNode
from classifiers.classifier import Classifier
from classifiers.perceptron import Perceptron
from classifiers.C45 import C45
from dataBalancing import DataBalance

class DTree(object):
	"""DTree class to store tree structure"""
	
	def __init__(self, num_classes, num_child, max_depth, data_type, data_dimension, data_balance, decision_type):
		"""
		Arguments:
		num_classes: Number of classes in data
		num_child:	Numer of child nodes per decision node
		max_depth:	Maximum depth of decision tree
		data_type:	One of {'numeric','image'}
		data_dimension:	Number of features in data; int for numeric , tuple of ints for image
		data_balance: Bool (whether to use data_balancing)
		decision_type: Classifier to be used
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
		
		#if self.data_type == 'numeric' and self.num_classes>2:
		#	self.decision_type = Perceptron
		#else:
		#	raise NotImplementedError('Feature not implemented')
		
		

	def train(self, data_file, epochs_per_node, batch_size, model_save_path):
		"""
		Build tree and save node parameters
		Arguments:
		data_file:	Data file for root node. Other nodes create their own file
		epochs_per_node: Number of training epochs for each node
		batch_size:	Batch size for training and predictions
		model_save_path:	Directory to save node parameters
		"""
		base = os.path.split(data_file)[0]
		node_to_process = 0
		new_node = DTNode(node_id=0, parent_id=0, node_depth=0, num_classes=self.num_classes,
		 num_child=self.num_child, data_file=data_file)
		self.nodes.append(new_node)
		while True:
			try:
				curr_node = self.nodes[node_to_process]
			except IndexError:
				print "{} nodes processed. Tree building done.".format(len(self.nodes))
				break
			curr_node.set_decision_maker(self.decision_type(input_dim=self.data_dimension, output_dim=self.num_child,
				 num_classes=self.num_classes, epochs=epochs_per_node, batch_size=batch_size))
			curr_node.set_child_id_start(len(self.nodes))
			if self.data_balance:
				db = DataBalance(os.path.join(base,'data_{}.csv'.format(i)) )
				db.data_balance(os.path.join(base,'b_data_{}.csv'.format(i)))
				# databalance(input=os.path.join(base,'data_{}.csv'.format(i)), output=os.path.join(base,'b_data_{}.csv'.format(i)))
				
			child_list = curr_node.train()
			curr_node.save_node_params(model_save_path)

			if child_list == [-1]:
				continue
			
			for i in child_list:
				# stop tree growth at max_depth
				num_child = self.num_child if self.max_depth>1+curr_node.node_depth else 0
				# if not using data balancing, send original file as balanced file
				balance_filename = 'b_data_{}.csv'.format(i) if self.data_balance else 'data_{}.csv'.format(i) 
				
				new_node = DTNode(node_id=i, parent_id=curr_node.node_id, node_depth=1+curr_node.node_depth,
				 num_classes=self.num_classes, num_child=num_child,
				  data_file=os.path.join(base,'data_{}.csv'.format(i)), balanced_file=os.path.join(base,balance_filename) )
				
				self.nodes.append(new_node)				
			
			node_to_process += 1
		self.built = True

	def save(self, model_save_file):
		"""
		Save tree structure. NOTE: Node parameters are saved separately
		Arguments:
		model_save_file:	File to save tree structure
		"""
		structure = {}
		for i in self.nodes:
			structure[i.node_id] = i.child_id
		with open(os.path.join(model_save_file), 'wb') as savefile:
			pickle.dump(structure, savefile, protocol=pickle.HIGHEST_PROTOCOL)

	def load_tree(self, model_save_file, model_save_path):
		structure = {}
		with open(os.path.join(model_file), 'rb') as modelfile:
			structure = pickle.load(f)

		node_to_process = 0
		new_node = DTNode(node_id=0, parent_id=0, node_depth=0, num_classes=self.num_classes,
		 num_child=self.num_child)
		self.nodes.append(new_node)
		while True:
			try:
				curr_node = self.nodes[node_to_process]
			except IndexError:
				print "{} nodes processed. Tree building done.".format(len(self.nodes))
				break
			curr_node.set_decision_maker(self.decision_type(input_dim=self.data_dimension, output_dim=self.num_child,
				 num_classes=self.num_classes, epochs=epochs_per_node, batch_size=batch_size))
			curr_node.load_node_params(model_save_path)
			
			child_list = structure[curr_node.node_id]

			if child_list == [-1]:
				continue
			
			for i in child_list:
				# stop tree growth at max_depth
				num_child = self.num_child if self.max_depth>1+curr_node.node_depth else 0
				# if not using data balancing, send original file as balanced file
				
				new_node = DTNode(node_id=i, parent_id=curr_node.node_id, node_depth=1+curr_node.node_depth,
				 num_classes=self.num_classes, num_child=num_child)			
				self.nodes.append(new_node)				
			
			node_to_process += 1

	def predict(self, model_file, model_save_path, data_file):
		self.load_tree(model_file, model_save_path)
		df = pd.read_csv(self.data_file, names=['features'])

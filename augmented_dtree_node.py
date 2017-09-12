from classifiers.classifier import Classifier

class DTNode(object):
	"""DTNode class to define DTree decisions"""
	def __init__(self, node_id, parent_id, node_depth, num_classes, num_child):
		super(DTNode, self).__init__()
		self.node_id = node_id
		self.parent_id = parent_id
		self.child_id = []
		self.node_depth = node_depth
		self.is_decision_node = False
		self.label = None
		self.num_classes = num_classes
		self.num_child = num_child
		
		self.decision_maker = None

		self.params = {}
		self.trained = False

	"""Set the decision maker for the current node"""
	def set_decision_maker(self, decision_maker):
		if not isinstance(decision_maker, Classifier):
			raise TypeError("DTNode received decision_maker of type: {}".format(type(decision_maker)))
		self.decision_maker = decision_maker

	"""Train on data and save params in Node"""
	def train(self, data_file):
		self.decision_maker.build()
		self.params = self.decision_maker.train(data_file)

	"""Save node params to file"""
	def save_node_params(self):
		# TODO
		pass

	"""Load node paramaters from file"""
	def load_node_params(self):
		# TODO
		pass

	"""Predict on file. If not a label node, returns relevant child node"""
	def predict(self):
		# TODO
		pass
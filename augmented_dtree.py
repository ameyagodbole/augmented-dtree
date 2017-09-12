class DTree(object):
	"""DTree class to store tree structure"""
	def __init__(self, num_classes, num_child, max_nodes, max_depth, data_type):
		super(DTree, self).__init__()
		self.num_classes = num_classes
		self.num_child = num_child
		self.max_nodes = max_nodes
		self.max_depth = max_depth
		self.data_type = data_type
		self.data_dimension = data_dimension
		self.nodes = []
		self.built = False
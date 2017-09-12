class Classifier(object):
	"""Definition for Classifier"""
	def __init__(self):
		super(Classifier, self).__init__()
	
	def build(self):
		"""Build classifier graph"""
		raise NotImplementedError

	def train(self, data_file):
		"""Train on data and return params"""
		raise NotImplementedError
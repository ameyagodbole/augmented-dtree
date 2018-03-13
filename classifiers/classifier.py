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

	def is_label(self):
		"""Checks if the data should be split or declared a leaf node"""
		raise NotImplementedError
		
	def max_freq(self):
		"""Returns dominant class"""
		raise NotImplementedError

	def predict(params, data):
		"""Returns which child data should go to"""
		raise NotImplementedError

import numpy as np
import logging
import pandas as pd
import time
from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as hcluster
# TODO: Switch to sklearn hierarchical


'''
def timing(f):
	def wrap(*args):
		time1 = time.time()
		ret = f(*args)
		time2 = time.time()
		print ('%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0))
		return ret
	return wrap
'''

class DataBalance(object):
	"""DataBalance class for data balancing"""
	
	def __init__(self, input_file, num_classes, mode):
		"""
		Arguments:
		input_file: File containing data to be balanced
		"""
		super(DataBalance, self).__init__()
		self.input_file = input_file
		self.data = None
		self.classFreq = []
		self.majClass = None
		self.size_class = None
		self.features = None
		self.num_classes = num_classes
		self.mode = mode


	def cluster(self, method):
		"""
		Clusters the data of each class.
		---
		Arguments:
		method : 'kmeans','hac'
		"""
		df = self.data
		for i in range(self.num_classes):
			data = df.loc[df['label'] == i, self.features].as_matrix()
			# TODO: Find permanent fix for
			# ValueError: The number of observations cannot be determined on an empty distance matrix
			if len(data)>1:	
				if method == 'kmeans':
					n_clusters = int(0.2*len(data))
					if n_clusters == 0:
						# TODO: Handle case better
						df.loc[df['label']==i, 'cluster'] = 0
						return
					kmeans_obj = KMeans(n_clusters=n_clusters, n_init=5, n_jobs=-1)
					kmeans = kmeans_obj.fit(data)
					df.loc[df['label']==i, 'cluster'] = kmeans.labels_
				elif method == 'hac':
					thresh = 10
					clusters = hcluster.fclusterdata(data, thresh, criterion='maxclust', method='average')
					df.loc[df['label']==i, 'cluster'] = clusters.astype(np.int32)
			elif len(data) == 1:
				df.loc[df['label']==i, 'cluster'] = 0
			else:
				pass


	def cbo_oversample(self, label, cluster, size):
		"""
		Oversamples the cluster 'cluster' of class 'label' to the required size.
		---
		Arguments:
		label:	Class to be oversampled
		cluster:	Cluster of class to be oversampled
		size: 	Required size to be sampled to
		"""
		df = self.data
		dfTemp = df.loc[(df['label'] == label) & (df['cluster'] == cluster)]

		# TODO: Find permanent fix
		if len(dfTemp) <= 1:
			return pd.DataFrame(columns=df.columns)

		s = size - dfTemp.shape[0]
		df2 = None

		if( (s > 0) & ((dfTemp.shape[0]) > 0)):
			dfRandom = dfTemp.sample(n = s, replace = True)
			dataTemp = dfTemp[self.features].as_matrix()
			mean = np.array(np.mean(dataTemp , axis = 0))
			allnewdata = []
			a = dfRandom[self.features]
			frac = [np.random.random() for i in range(dfRandom.shape[0])]
			b = np.asarray([[(1-f) for row in range(len(self.features))]for f in frac]) * np.array(a)
			c = np.asarray([[f for row in range(len(self.features))] for f in frac])* np.asarray([mean for m in range(dfRandom.shape[0])])
			newdata = b+c
			newdata = newdata.tolist()
			df2 = pd.DataFrame(newdata,columns=self.features)
			df2['label'] = label
			df2['cluster'] = cluster
			df2['original'] = 0			

		return df2


	def cbo_get_params(self):
		"""
		Calculates parameters majClass, size_class
		"""
		df = self.data

		self.classFreq = []
		for i in range(self.num_classes):
			self.classFreq.append(len(df.loc[df['label'] == i]))

		self.majClass = self.classFreq.index(max(self.classFreq))

		dfTemp = df.loc[df['label'] == self.majClass]

		clusterFreq = []
		for i in np.unique(dfTemp['cluster']):
			clusterFreq.append(len(dfTemp.loc[dfTemp['cluster'] == i]))

		maxCluster = max(clusterFreq)

		self.size_class = maxCluster * (int(len(np.unique(dfTemp['cluster']))))


	def hac_oversample(self):
		"""
		Oversample by hierarchical clustering
		"""
		self.cluster('hac')
		oversample = self.cbo_oversample
		df = self.data
		self.cbo_get_params()

		for i in range(self.num_classes):
			dfTemp = df.loc[(df['label'] == i)]
			num_clusters = (int(len(np.unique(dfTemp['cluster']))))
			print('Balancing Class {}'.format(i))

			for j in np.unique(dfTemp['cluster']):
				df = pd.concat([df,oversample(i, j, int(np.ceil(float(self.size_class)/num_clusters)))],ignore_index=False)

		return df


	def kmeans_oversample(self):
		"""
		Oversample by KMeans clustering
		"""
		self.cluster('kmeans')
		oversample = self.cbo_oversample
		df = self.data
		self.cbo_get_params()

		for i in range(self.num_classes):
			dfTemp = df.loc[(df['label'] == i)]
			num_clusters = (int(len(np.unique(dfTemp['cluster']))))
			print('Balancing Class {}'.format(i))

			for j in np.unique(dfTemp['cluster']):
				df = pd.concat([df,oversample(i, j, int(np.ceil(float(self.size_class)/num_clusters)))],ignore_index=False)

		return df


	def kmeans_undersample(self):
		'''
		Undersample majority class with its centroids
		'''
		df = self.data

		cc = ClusterCentroids(n_jobs=-1)
		data = df[self.features].as_matrix()
		labels = df['label']
		data_resampled, label_resampled = cc.fit_sample(data, labels)

		df2 = pd.DataFrame(data_resampled.tolist(),columns=self.features)

		df2['label'] = label_resampled
		df2['cluster'] = 0
		df2['original'] = 0

		return df2


	def data_balance(self, out_file_name):
		"""
		Balances the data and saves it in a csv file.
		Arguments:
		name: 	name of csv file to save to
		"""
		self.load()
		if self.data.empty or len(self.data)<=1:
			return
		dfBalanced = None
		if self.mode == 'hac_os_no_us':
			dfBalanced = self.hac_oversample()
		elif self.mode == 'kmeans_os_no_us':
			dfBalanced = self.kmeans_oversample()
		elif self.mode == 'no_os_kmeans_us':
			dfBalanced = self.kmeans_undersample()
		else:
			raise NotImplementedError

		out_csv = dfBalanced[self.features+['label']]
		out_csv.to_csv(out_file_name,index=False)


	def load(self):
		"""
		Loads data from csv file to a dataframe
		"""
		df = pd.read_csv(self.input_file)
		self.features = [col for col in df.columns if col!='label']

		df['cluster'] = np.nan	
		df['original'] = 1
		self.data = df


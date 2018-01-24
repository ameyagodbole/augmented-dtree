import numpy as np
import logging
# import scipy.cluster.hierarchy as hcluster
# import scipy.cluster
# import scipy.stats
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import time
from imblearn.under_sampling import TomekLinks,EditedNearestNeighbours

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
	
	def __init__(self, input_file, num_classes):
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
		self.k_means = {}

	def cluster(self):
		"""
		Clusters the data of each class.
		"""
		
		df = self.data
		thresh = 100
		for i in range(self.num_classes):
			data = df.loc[df['label'] == i, self.features].as_matrix()
			if len(data)>2:
				km = KMeans(n_clusters=max(int(0.2*len(data)),2))
				clusters = km.fit_predict(data)
				k_means[i] = km.cluster_centers_
				df.loc[df['label']==i, 'cluster'] = np.ravel(clusters).astype(np.int32)
			else:
				df.loc[df['label']==i, 'cluster'] = 0
	

	def oversample(self, label, cluster, size):
		"""
		Oversamples the cluster 'cluster' of class 'label' to the required size.
		Arguments:
		label:	Class to be oversampled
		cluster:	Cluster of class to be oversampled
		size: 	Required size to be sampled to
		"""
		df = self.data
		dfTemp = df.loc[(df['label'] == label) & (df['cluster'] == cluster)]
	
		s = size - dfTemp.shape[0]
		df2 = pd.DataFrame(columns=self.features+['label','cluster','original'])

		if( (s > 0) & ((dfTemp.shape[0]) > 0)):
			dfRandom = dfTemp.sample(n = s, replace = True)
			dataTemp = dfTemp[self.features].as_matrix()
			try:
				mean = 	self.k_means[label][cluster,:]
			except KeyError:
				return df2
			except IndexError:
				return df2
			mean = np.reshape(mean, (1,-1))
			allnewdata = []
			a = dfRandom.as_matrix(self.features)
			frac = np.random.uniform(size=(dfRandom.shape[0],1))
			b = frac * a
			c = (1.-frac)* mean
			newdata = b+c
			df2 = pd.DataFrame(newdata,columns=self.features)
			df2['label'] = label
			df2['cluster'] = cluster
			df2['original'] = 0
			
		return df2

	def get_params(self):
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



	def balance(self):
		"""
		Increases the size of every class to that of the balanced majority class
		"""
		oversample = self.oversample
		df = self.data
		self.get_params()

		for i in range(self.num_classes):
			dfTemp = df.loc[(df['label'] == i)]
			
			ratio = float(len(dfTemp))/len(df)
			if ratio < 0.05:
				continue

			num_clusters = (int(len(np.unique(dfTemp['cluster']))))
			print('Balancing Class {}'.format(i))

			for j in np.unique(dfTemp['cluster']):
				
				lenj = dfTemp.loc[dfTemp['cluster'] == j].shape[0]
				'''
				if lenj/len(dfTemp)<0.05:
					continue
				'''
				df = pd.concat([df,oversample(i, j, int(np.ceil(float(self.size_class)/num_clusters)))],ignore_index=True)
		return df

	def undersample(self, df):
		n_neighbors = 6
		knn = NearestNeighbors(n_neighbors=n_neighbors)
		dfsampled = df.loc[df['original']==0]
		knn.fit(df.as_matrix(self.features))
		query = knn.kneighbors(dfsampled.as_matrix(self.features), return_distance=False)
		labels = np.ravel(df.as_matrix('label'))
		# nearest_labels = np.reshape(labels[query],(len(dfsampled),-1))
		to_drop=[]
		for i,index in enumerate(dfsampled.index):
			(values,counts) = np.unique(labels[query[i,...]],return_counts=True)
			ind=np.argmax(counts)
			if(values[ind]!=row['label']):
				to_drop.append(index)
		dfsampled.drop(to_drop, inplace=True)
		return pd.concat([df.loc[df['original']==1],dfsampled],ignore_index=True)

	def data_balance(self, out_file_name):
		"""
		Balances the data and saves it in a csv file.
		Arguments:
		name: 	name of csv file to save to
		"""
		self.load()
		if self.data.empty:
			return
		self.cluster()
		dfBalanced = self.balance()
		print 'Oversampled size:',len(dfBalanced)
		"""==============================="""
		# X = out_csv.as_matrix(self.features)
		# y = np.ravel(out_csv.as_matrix(['label']))

		# tl = TomekLinks(ratio='all',random_state=42,n_jobs=4)
		# X_res, y_res = tl.fit_sample(X, y)
		
		# enn = EditedNearestNeighbours(ratio='all',kind_sel='mode',n_neighbors=5,random_state=42,n_jobs=4)
		# X_res, y_res = enn.fit_sample(X, y)

		# out_csv = pd.DataFrame(X_res, columns=self.features)
		# out_csv['label'] = y_res
		
		dfBalanced = self.undersample(dfBalanced)
		print 'Undersampled size:',len(dfBalanced)
		"""==============================="""
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
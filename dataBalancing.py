import numpy as np
import logging
import scipy.cluster.hierarchy as hcluster
import scipy.cluster
import scipy.stats
import pandas as pd
from sklearn.cluster import KMeans

import time
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

	def cluster(self):
		"""
		Clusters the data of each class.
		"""
		
		df = self.data
		thresh = 100
		for i in range(self.num_classes):
			data = df.loc[df['label'] == i, self.features].as_matrix()
			# if len(data)>2:	
			# 	clusters = hcluster.fclusterdata(data, thresh, criterion='distance', method='average')
			# 	df.loc[df['label']==i, 'cluster'] = clusters.astype(np.int32)
			# else:
			# 	df.loc[df['label']==i, 'cluster'] = 0
			if len(data)>5:
				kmeans = KMeans(n_clusters=int(0.2*len(data)), random_state=0).fit(data)
				clusters = kmeans.labels_
				#clusters = hcluster.fclusterdata(data, thresh, criterion='distance', method='average')
				# print(clusters)
				df.loc[df['label']==i, 'cluster'] = clusters.astype(np.int32)
			else:
				df.loc[df['label']==i, 'cluster'] = 0
				
	def cluster_hybrid(self):
		"""
		Clusters the data of each class.
		"""
		
		df = self.data
		thresh = 1
		minority = []
		for i in range(self.num_classes):
			data = df.loc[df['label'] == i, self.features].as_matrix()
			try:
				if len(data)>0.05*len(df):	
					if len(data)>5:
						kmeans = KMeans(n_clusters=int(0.2*len(data)), random_state=0).fit(data)
						clusters = kmeans.labels_
						#clusters = hcluster.fclusterdata(data, thresh, criterion='distance', method='average')
						# print(clusters)
						df.loc[df['label']==i, 'cluster'] = clusters.astype(np.int32)
					else:
						df.loc[df['label']==i, 'cluster'] = 0
				else:
					df.loc[df['label']==i, 'cluster'] = -1
					minority.append(i)
			except ValueError:
				continue

		return minority
	

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
			'''

			for k,r in dfRandom.iterrows():
				a = r[self.features]
				frac = np.random.random()
				newdata = (1-frac) * np.array(a) + frac * mean
				allnewdata.append(newdata.tolist() + [label,cluster,0])
			df2 = pd.DataFrame(allnewdata,columns=self.features+['label','cluster','original'])
			'''
			#df = pd.concat([df,df2],ignore_index=False)
			

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
			#print(i)
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
			
			ratio = len(dfTemp)/len(df)
			if ratio < 0.01:
				continue
			

			num_clusters = (int(len(np.unique(dfTemp['cluster']))))
			logging.debug('Balancing Class {}'.format(i))

			for j in np.unique(dfTemp['cluster']):
				
				lenj = dfTemp.loc[dfTemp['cluster'] == j].shape[0]
				'''
				if lenj/len(dfTemp)<0.05:
					continue
				'''
				df = pd.concat([df,oversample(i, j, int(np.ceil(float(self.size_class)/num_clusters)))],ignore_index=False)
		return df
	def balance_hybrid(self,minority):
		"""
		Increases the size of every class to that of the balanced majority class
		"""
		oversample = self.oversample
		df = self.data
		self.get_params()

		for i in range(self.num_classes):
			if i in minority:
				df_drop = df[ df['label'] ==i ]
				df = df.drop(df_drop.index, axis=0)
				continue
			dfTemp = df.loc[(df['label'] == i)]
			
			ratio = float(len(dfTemp))/len(df)
			if ratio < 0.01:
				continue
			

			num_clusters = (int(len(np.unique(dfTemp['cluster']))))
			logging.debug('Balancing Class {}'.format(i))

			for j in np.unique(dfTemp['cluster']):
				
				lenj = dfTemp.loc[dfTemp['cluster'] == j].shape[0]
				'''
				if lenj/len(dfTemp)<0.05:
					continue
				'''
				df = pd.concat([df,self.oversample(i, j, int(np.ceil(float(self.size_class)/num_clusters)))],ignore_index=False)
		return df


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
		out_csv = dfBalanced[self.features+['label']]
		out_csv.to_csv(out_file_name,index=False)

	def data_balance_hybrid(self, out_file_name):
		"""
		Balances the data and saves it in a csv file.
		Arguments:
		name: 	name of csv file to save to
		"""
		self.load()
		if self.data.empty:
			return
		minority = self.cluster_hybrid()
		dfBalanced = self.balance_hybrid(minority)
		out_csv = dfBalanced[self.features+['label']]
		out_csv.to_csv(out_file_name,index=False)
		return minority


	def load(self):
		"""
		Loads data from csv file to a dataframe
		"""
		df = pd.read_csv(self.input_file)
		self.features = [col for col in df.columns if col!='label']

		df['cluster'] = np.nan	
		df['original'] = 1
		self.data = df
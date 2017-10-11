

import numpy as np
import scipy.cluster.hierarchy as hcluster
import scipy.cluster
import scipy.stats
import pandas as pd

class DataBalance(object):
	"""DataBalance class for data balancing"""
	
	def __init__(self, input_file):
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


	def cluster(self):
		"""
		Clusters the data of each class.
		"""
		thresh = 1
		df = self.data
		for i in range(max(df['label'])+1):

			data = np.array(df.loc[df['label'] == i, self.features].tolist())		
			clusters = hcluster.fclusterdata(data, thresh, criterion='distance',method='average')
			df.loc[df['label']==i, 'cluster'] = clusters
				
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

		if( (s > 0) & ((dfTemp.shape[0]) > 0)):
			dfRandom = dfTemp.sample(n = s, replace = True)
			dataTemp = np.array( dfTemp[self.features].tolist())
			mean = np.mean(dataTemp , axis = 0)

			for k,r in dfRandom.iterrows():
				a = r[self.features]
				newdata = (np.array(a) + np.array(mean)) / 2
				df2 = pd.DataFrame([[newdata,i,j,0]],columns=[self.features,'label','cluster','original'])
				df = pd.concat([df,df2])

		return df

	def get_params(self):
		"""
		Calculates parameters majClass, size_class
		"""
		df = self.data
		
		for i in range(max(df['label'])+1):
			self.classFreq.append(df.loc[df['label'] == i].shape[0])

		self.majClass = self.classFreq.index(max(self.classFreq))

		dfTemp = df.loc[(df['label'] == self.majClass)]

		clusterFreq = []
		for i in range(int(max(dfTemp['cluster']))+1):
			clusterFreq.append(dfTemp.loc[dfTemp['cluster'] == i].shape[0])

		maxCluster = max(clusterFreq)

		self.size_class = maxCluster * (int(max(dfTemp['cluster']))+1)



	def balance(self):
		"""
		Increases the size of every class to that of the balanced majority class
		"""
		
		df = self.data
		self.get_params()

		for i in range(max(df['label'])+1):
			dfTemp = df.loc[(df['label'] == i)]
			num_clusters = (int(max(dfTemp['cluster']))+1)

			for j in range(1, int(max(dfTemp['cluster']))+1):
				df = oversample(df, i, j, self.size_class/num_clusters)

		#print(df.shape)
		return df


	def data_balance(self, out_file_name):
		"""
		Balances the data and saves it in a csv file.
		Arguments:
		name: 	name of csv file to save to

		"""
		self.load()
		self.cluster()
		dfBalanced = self.balance()
		out_csv = dfBalanced[[self.features,'label']]
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


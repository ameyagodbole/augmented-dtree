import numpy as np
import logging
import scipy.cluster.hierarchy as hcluster
import scipy.cluster
import scipy.stats
import pandas as pd
import time
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks,EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from sklearn.neighbors import NearestNeighbors

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
			if len(data)>2:	
				clusters = hcluster.fclusterdata(data, thresh, criterion='distance', method='average')
				df.loc[df['label']==i, 'cluster'] = clusters.astype(np.int32)
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

	def oversample_knn(self, label, cluster, size):
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
			if dataTemp.shape[0]>6:
				nn = 5
			elif dataTemp.shape[0]==1:
				return None
			else:
				nn = dataTemp.shape[0]-1
			nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(dfTemp[self.features])
			a = dfRandom[self.features]
			#dfRandom = nbrs.sample(n = s, replace = True)
			distances, indices = nbrs.kneighbors(a)
			
			#frac = [np.random.random() for i in range(dfRandom.shape[0])]
			#b = np.asarray([[(1-f) for row in range(len(self.features))]for f in frac]) * np.array(a)
			#c = np.asarray([[f for row in range(len(self.features))] for f in frac])* np.asarray([mean for m in range(dfRandom.shape[0])])
			#newdata = b+c
			newdata = np.asarray([np.mean([dataTemp[i] for i in indices[j]],axis = 0)for j in range(dfRandom.shape[0])])
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
				df = pd.concat([df,self.oversample_knn(i, j, int(np.ceil(float(self.size_class)/num_clusters)))],ignore_index=False)
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
		"""==============================="""
		out_temp = pd.DataFrame()
		dropped = pd.DataFrame()
		for c in np.unique(out_csv['label']):
			if len(out_csv.loc[out_csv['label']==c])>6:
				out_temp = pd.concat([out_temp,out_csv.loc[out_csv['label']==c]], ignore_index = True)
			else:
				dropped = pd.concat([dropped,out_csv.loc[out_csv['label']==c]], ignore_index = True)
		print 'Original size:',len(self.data)
		print 'Oversampled size:',len(out_csv)
		X = out_temp.as_matrix(self.features)
		y = np.ravel(out_temp.as_matrix(['label']))

		#tl = TomekLinks(ratio='all',random_state=42,n_jobs=4)
		#X_res, y_res = tl.fit_sample(X, y)
		
		if len(np.unique(y))>1:
			#sm = SMOTE(ratio='all',n_jobs=4)
			enn = EditedNearestNeighbours(ratio='all',kind_sel='mode',n_neighbors=5,random_state=42,n_jobs=4)
			#smenn = SMOTEENN(ratio='all',smote=sm,enn=enn)
			X_res, y_res = enn.fit_sample(X, y)

			out_csv = pd.DataFrame(X_res, columns=self.features)
			out_csv['label'] = y_res
			out_csv = pd.concat([out_csv,dropped], ignore_index = True)
		print 'Undersampled size:',len(out_csv)
		"""==============================="""
		out_csv.to_csv(out_file_name,index=False)

	def data_balance_CBO_enn(self, out_file_name):
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
		"""==============================="""
		#out_csv = self.data
		print 'Original size:',len(out_csv)
		X = out_csv.as_matrix(self.features)
		y = np.ravel(out_csv.as_matrix(['label']))
		
		enn = EditedNearestNeighbours(ratio='all',kind_sel='mode',n_neighbors=5,random_state=42,n_jobs=4)
		X_res, y_res = enn.fit_sample(X, y)

		out_csv = pd.DataFrame(X_res, columns=self.features)
		out_csv['label'] = y_res
		print 'Undersampled size:',len(out_csv)
		"""==============================="""
		out_csv.to_csv(out_file_name,index=False)

	def data_balance_CBO(self, out_file_name):
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
		"""==============================="""
		print 'Undersampled size:',len(out_csv)
		"""==============================="""
		out_csv.to_csv(out_file_name,index=False)

	def load(self):
		"""
		Loads data from csv file to a dataframe
		"""
		df = pd.read_csv(self.input_file)
		self.features = [col for col in df.columns if col!='label']

		# df['cluster'] = np.nan	
		# df['original'] = 1
		self.data = df

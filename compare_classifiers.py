import pickle
from glob import glob
from augmented_dtree import DTree
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import os
import shutil
import time
from dataBalancing import DataBalance
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks,EditedNearestNeighbours
from imblearn.combine import SMOTEENN



def preprocess(add_header,data_file, test, num_features, lab_beginning,normalize, class_format, num_classes):
	dataset = None
	if add_header:
		dataset = pd.read_csv(data_file, header = None)	
	else:
		dataset = pd.read_csv(data_file)
	
	if add_header:
		col_list = []

		if lab_beginning==True:
			col_list.append('label')

		for i in range(num_features):
			col_list.append('f{}'.format(i))

		if(lab_beginning==False):
			col_list.append('label')
	
		dataset.columns = col_list
		#print(dataset.columns)
	if test == True:
		dataset['assigned_node'] = 0

	if normalize:
		col = dataset['label']
		dataset = (dataset - dataset.mean(axis=0))/dataset.std(axis=0)
		dataset['label'] = col

	if class_format:
		if pd.np.sum(df['label']==num_classes):
			df['label'] -= 1
	dataset.to_csv(data_file, index = False)

def perceptron_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size):
	
	model_save_path = os.path.join(os.path.join(data_path,'perceptron_tree'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_Perceptron_tree'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_Perceptron_tree'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)
def perceptron_treeWDB(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size):
	
	model_save_path = os.path.join(os.path.join(data_path,'perceptron_tree'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_Perceptron_treeWDB'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_Perceptron_treeWDB'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)

def decision_tree(data_path, train_file, test_file, output_file, max_depth, count_threshold):
	os.path.join(data_path, train_file)
	dataset = pd.read_csv(os.path.join(data_path, train_file))
	features = [col for col in dataset.columns if col!='label']
	X = dataset.as_matrix(features)
	Y = pd.np.ravel(dataset.as_matrix(['label']))
	clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, min_samples_split=count_threshold)
	clf = clf.fit(X, Y)
	print('training done')
	
	print(clf.tree_.node_count)
	with open(os.path.join(data_path,'C45','model.pkl'),'w') as file:
		pickle.dump(clf, file, protocol=pickle.HIGHEST_PROTOCOL)

	test = pd.read_csv(os.path.join(data_path, test_file))
	cols = [col for col in test.columns if col not in ['predicted_label', 'label', 'assigned_node']]
	
	x = test.as_matrix(cols)
	pred = clf.predict(x)
	#print(pred)

	test['predicted_label'] = pred
	test = test[['label','predicted_label']]
	test.to_csv(os.path.join(os.path.join(data_path, 'C45'), output_file), index = False)
	cnt = 0
	acc = pd.np.sum(test['label']==test['predicted_label']).astype(pd.np.float32) / len(test)
	print("Accuracy: {}".format(acc))
	f = open(os.path.join(os.path.join(data_path, 'C45'),'accuracy.txt'), 'w')
	f.write("Accuracy: {}".format(acc))
	f.close()

def accuracy_class(df):
	acc = []
	for c in np.unique(df['label']):
		tot = len(df[df.label == c])
		df1 = df[df.label == c]
		corr = len(df1[df1.predicted_label == c ])
		acc.append(float(corr)/tot)
	return acc



def svm(data_path, train_file, test_file, output_file, type_file):
	os.path.join(data_path, train_file)
	dataset = pd.read_csv(os.path.join(data_path, train_file))
	features = [col for col in dataset.columns if col!='label']
	X = dataset.as_matrix(features)
	Y = pd.np.ravel(dataset.as_matrix(['label']))
	clf = SVC(verbose = True)
	start_time = time.clock()
	clf = clf.fit(X, Y)
	print('training done')
	print time.clock() - start_time, "seconds for training"
	f = open(os.path.join(data_path, 'SVM',type_file+'time.txt'), 'w')
	f.write('training time: {}'.format(time.clock() - start_time))
	
	
	print(clf.n_support_)
	with open(os.path.join(data_path,'SVM','model.pkl'),'w') as file:
		pickle.dump(clf, file, protocol=pickle.HIGHEST_PROTOCOL)
	
	test = pd.read_csv(os.path.join(data_path, test_file))
	cols = [col for col in test.columns if col not in ['predicted_label', 'label', 'assigned_node']]
	
	x = test.as_matrix(cols)
	start_time = time.clock()
	pred = clf.predict(x)
	#print(pred)
	print time.clock() - start_time, "seconds for testing"
	f.write('testing time: {}'.format(time.clock() - start_time))
	f.close()
	test['predicted_label'] = pred
	test = test[['label','predicted_label']]
	test.to_csv(os.path.join(data_path, 'SVM', type_file + output_file), index = False)
	acc = pd.np.sum(test['label']==test['predicted_label']).astype(pd.np.float32) / len(test)
	print("Accuracy: {}".format(acc))
	f = open(os.path.join(data_path, 'SVM',type_file+'accuracy.txt'), 'w')

	f.write("Accuracy: {}\n".format(acc))
	f.write("var: {}\n".format(np.var(accuracy_class(test))))
	f.write("min: {}\n".format(np.min(accuracy_class(test))))
	f.write("max: {}\n".format(np.max(accuracy_class(test))))

	f.close()

def mlp(data_path, train_file, test_file, output_file, num_classes, type_file):

	os.path.join(data_path, train_file)
	dataset = pd.read_csv(os.path.join(data_path, train_file))
	features = [col for col in dataset.columns if col!='label']
	X = dataset.as_matrix(features)
	Y = pd.np.ravel(dataset.as_matrix(['label']))
	mlp = MLPClassifier(hidden_layer_sizes=(num_classes+len(features),), max_iter=500, alpha=1e-4,
					solver='sgd', verbose=10, tol=1e-4, random_state=1,
					learning_rate_init=.1)
	start_time = time.clock()
	mlp.fit(X, Y)
	print('training done for '+ train_file)
	f = open(os.path.join(data_path, 'MLP',type_file+'time.txt'), 'w')
	print time.clock() - start_time, "seconds for training"
	f.write('training time: {}'.format(time.clock() - start_time))
	test = pd.read_csv(os.path.join(data_path, test_file))
	cols = [col for col in test.columns if col not in ['predicted_label', 'label', 'assigned_node']]
	
	x = test.as_matrix(cols)
	start_time = time.clock()
	pred = mlp.predict(x)
	#print(pred)
	print time.clock() - start_time, "seconds for testing"
	f.write('testing time: {}'.format(time.clock() - start_time))
	f.close()
	test['predicted_label'] = pred
	test = test[['label','predicted_label']]
	test.to_csv(os.path.join(data_path, 'MLP', type_file + output_file), index = False)
	acc = pd.np.sum(test['label']==test['predicted_label']).astype(pd.np.float32) / len(test)
	print("Accuracy: {}".format(acc))
	f = open(os.path.join(data_path, 'MLP',type_file+'accuracy.txt'), 'w')
	f.write("Accuracy: {}\n".format(acc))
	f.write("var: {}\n".format(np.var(accuracy_class(test))))
	f.write("min: {}\n".format(np.min(accuracy_class(test))))
	f.write("max: {}\n".format(np.max(accuracy_class(test))))
	f.close()
	
def C45_treeWDB(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size):
	
	model_save_path = os.path.join(os.path.join(data_path,'C45'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False, decision_type = 'C45', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_C45_treeWDB'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False, decision_type = 'C45', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_C45_treeWDB'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)

def C45_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size):
	
	model_save_path = os.path.join(os.path.join(data_path,'C45'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True, decision_type = 'C45', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_C45_tree'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True, decision_type = 'C45', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_C45_tree'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)

def databalance_enn(data_path, train_file, num_classes):
	dataset = pd.read_csv(os.path.join(data_path, train_file))
	
	out_csv = dataset
	features = [col for col in df.columns if col!='label']
	dropped = pd.DataFrame(columns = out_csv.columns)
	counts = [np.sum(out_csv['label']==c) for c in np.unique(out_csv['label'])]
	most_freq_class = np.unique(out_csv['label'])[np.argmax(counts)] 

	for c in np.unique(dataset['label']):
		if np.sum(dataset['label']==c)<6:
			dropped = pd.concat([dropped,out_csv.loc[out_csv['label']==c]], ignore_index = True)
			out_csv = out_csv.loc[out_csv['label']!=c]

	print 'Original size:',len(out_csv)
	X = out_csv.as_matrix(features)
	y = np.ravel(out_csv.as_matrix(['label']))

	enn = EditedNearestNeighbours(ratio='all',kind_sel='mode',n_neighbors=5,random_state=42,n_jobs=4)
	X_res, y_res = enn.fit_sample(X, y)

	out_csv = pd.DataFrame(X_res, columns=features)
	out_csv['label'] = y_res
	out_csv = pd.concat([out_csv, dropped], ignore_index=True)
	print 'Resampled size:',len(out_csv)
	"""==============================="""
	out_csv.to_csv(os.path.join(data_path, 'balance_enn_'+ train_file),index=False)

def clean_dir_files(folder):
	for f in os.listdir(folder):
		ob = os.path.join(folder,f)
		if os.path.isdir(ob):
			clean_dir_files(ob)
		else:
			os.remove(ob)

if __name__ == "__main__":
	#all_data_sets = ['letterdata']
	all_data_sets = [ 'satimage','seismic','sensorless','shuttle','pendigits','protein','connect4', 'vowel', 'dna']
	for d_set in all_data_sets:
		data_path = './datasets/'+d_set
		train_file = os.path.basename(glob(os.path.join(data_path, d_set +'_train.csv'))[0])
		test_file = os.path.basename(glob(os.path.join(data_path,d_set+'_test.csv'))[0])
		output_file= 'prediction.csv'

		

		df = pd.read_csv(os.path.join(data_path,train_file))
		num_classes = len(pd.np.unique(df['label']))
		num_child = 2
		max_depth = 15
		data_type = 'float'
		data_dimension = len(df.columns) - 1

		#databalance_enn(data_path, train_file, num_classes)

		db = DataBalance(os.path.join(data_path, train_file) , num_classes)
		db.data_balance_CBO_enn(os.path.join(data_path, 'CBO_enn_balanced_'+train_file))
		db.data_balance_CBO(os.path.join(data_path, 'CBO_balanced_'+train_file))

		count_threshold = 2*num_classes*6
		purity_threshold = 0.95
		impurity_drop_threshold = 0.05

		epochs_per_node = 200
		batch_size = 500
		header = True #should headers be added to csv
		lab_beginning = True # do labels occur in the first col of csv
		normalize = False #should data be normalized
		class_format = True # set to true if classes are labeled 1 to n
		preprocess_needed = True #is preprocessing needed (Keep this true only for running the first time)

		'''
		print('Training mlp')
		mlp(data_path, train_file, test_file, output_file, num_classes, 'unbalanced')
		mlp(data_path, 'CBO_balanced_'+ train_file, test_file, output_file, num_classes, 'CBO_balanced_')
		mlp(data_path, 'CBO_enn_balanced_'+train_file, test_file, output_file, num_classes, 'CBO_enn_balanced_')
		
		print('Training svm')
		svm(data_path, train_file, test_file, output_file,  'unbalanced')
		svm(data_path, 'CBO_balanced_'+ train_file, test_file, output_file, 'CBO_balanced_')
		svm(data_path, 'CBO_enn_balanced_'+train_file, test_file, output_file,'CBO_enn_balanced_')
		'''

		#print('Training decision tree')
		#decision_tree(data_path, train_file, test_file, output_file)
		
		
		print('training perceptron tree with data balancing')
		perceptron_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_without_imp_thresh'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))
		
		print('training perceptron tree without data balancing')
		perceptron_treeWDB(data_path, train_file, test_file, 'prediction_wdb.csv', num_classes, num_child, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_without_imp_thresh_wdb'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))

		'''
		'''
		print('training perceptron tree with 3 child nodes')
		perceptron_tree(data_path, train_file, test_file, 'prediction_3child.csv', num_classes, 3, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_3c_db'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))

		print('training perceptron tree with 3 child nodes')
		perceptron_treeWDB(data_path, train_file, test_file, 'prediction_3childWDB.csv', num_classes, 3, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_3c_wdb'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))

		print('training perceptron tree with 4 child nodes')
		perceptron_tree(data_path, train_file, test_file, 'prediction_4child.csv', num_classes, 4, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_4c_db'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))

		print('training perceptron tree with 4 child nodes')
		perceptron_treeWDB(data_path, train_file, test_file, 'prediction_4childWDB.csv', num_classes, 4, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_4c_wdb'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))

		print('training perceptron tree with 5 child nodes')
		perceptron_tree(data_path, train_file, test_file, 'prediction_5child.csv', num_classes, 5, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_5c_db'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))

		print('training perceptron tree with 5 child nodes')
		perceptron_treeWDB(data_path, train_file, test_file, 'prediction_5childWDB.csv', num_classes, 5, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'perceptron_tree'),os.path.join(data_path,'perceptron_tree_5c_wdb'))
		clean_dir_files(os.path.join(data_path,'perceptron_tree'))
		
		
		print('training C45 tree without data balancing')
		C45_treeWDB(data_path, train_file, test_file, 'C45_wdb.csv', num_classes, num_child, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'C45'),os.path.join(data_path,'C45_tree_without_imp_thresh_wdb'))
		clean_dir_files(os.path.join(data_path,'C45'))

		print('training C45 tree with data balancing')
		C45_tree(data_path, train_file, test_file, 'C45_db.csv', num_classes, num_child, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		shutil.copytree(os.path.join(data_path,'C45'),os.path.join(data_path,'C45_tree_without_imp_thresh'))
		# clean_dir_files(os.path.join(data_path,'C45'))
		print('{} dataset processed'.format(d_set))
		print('+++++++++++++++++++++++++++++++++++++++')

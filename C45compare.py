import pickle
from glob import glob
from augmented_dtree import DTree
import pandas as pd
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import os
import shutil

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

def C45_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size):
	
	model_save_path = os.path.join(os.path.join(data_path,'C45_tree'),'model')
	
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

def C45_treeWDB(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size):
	
	model_save_path = os.path.join(os.path.join(data_path,'C45_tree'),'model')
	
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







if __name__ == "__main__":
	#all_data_sets = ['satimage','vowel','sensorless','shuttle','seismic','connect-4']
	all_data_sets = ['seismic']
	#all_data_sets = ['letterdata']
	# all_data_sets = ['pendigits']
	for d_set in all_data_sets:
		data_path = './datasets/'+d_set
		train_file = os.path.basename(glob(os.path.join(data_path,'*train.csv'))[0])
		test_file = os.path.basename(glob(os.path.join(data_path,'*test.csv'))[0])
		output_file= 'prediction.csv'
		df = pd.read_csv(os.path.join(data_path,train_file))
		num_classes = len(pd.np.unique(df['label']))
		data_dimension = len(df.columns) - 1
		del df
		
		num_child = 2
		max_depth = 50
		data_type = 'float'
		
		count_threshold = 10
		purity_threshold = 0.95
		impurity_drop_threshold = 0.05

		epochs_per_node = 200
		batch_size = 500

		
		print('training perceptron tree with data balancing')
		'''
		output_file= 'C45_balanced_prediction.csv'
		C45_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		if os.path.isdir(os.path.join(data_path,'C45_tree_2c_db')):
			shutil.rmtree(os.path.join(data_path,'C45_tree_2c_db'))
		shutil.copytree(os.path.join(data_path,'C45_tree'),os.path.join(data_path,'C45_tree_2c_db'))
		'''

		print "=========================================="
		print('training perceptron tree wdb without data balancing')
		output_file= 'C45_Notbalanced_prediction.csv'
		C45_treeWDB(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, 
			data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		if os.path.isdir(os.path.join(data_path,'C45_tree_2c_wdb')):
			shutil.rmtree(os.path.join(data_path,'C45_tree_2c_wdb'))
		shutil.copytree(os.path.join(data_path,'C45_tree'),os.path.join(data_path,'C45_tree_2c_wdb'))
		print "=========================================="
		
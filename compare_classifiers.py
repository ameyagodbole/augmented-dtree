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
		dataset = (dataset- dataset.min())/dataset.max()
		dataset['label'] = col
	#print(dataset.columns)
	if class_format:
		for ix,r in dataset.iterrows():
			if r['label']==num_classes:
				#print(r['label'])
				r['label'] = 0
	dataset.to_csv(data_file, index = False)

def perceptron_tree_hybrid(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size, balance_mode):
	
	model_save_path = os.path.join(os.path.join(data_path,'perceptron_tree_hybrid'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True, balance_mode = balance_mode, decision_type = 'Perceptron_hybrid', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_Perceptron_tree_hybrid'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True,balance_mode = balance_mode, decision_type = 'Perceptron_hybrid', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_Perceptron_tree_hybrid'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)

def perceptron_tree_hybridWDB(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size):
	
	model_save_path = os.path.join(os.path.join(data_path,'perceptron_tree_hybrid'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False, decision_type = 'Perceptron_hybrid', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_Perceptron_tree_hybridWDB'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False, decision_type = 'Perceptron_hybrid', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_Perceptron_tree_hybridWDB'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)

def perceptron_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size, balance_mode):
	
	model_save_path = os.path.join(os.path.join(data_path,'perceptron_tree'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True,balance_mode = balance_mode, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_Perceptron_tree'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = True,balance_mode = balance_mode, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_Perceptron_tree'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)

def perceptron_treeWDB(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, data_dimension, 
	 count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size, balance_mode):
	
	model_save_path = os.path.join(os.path.join(data_path,'perceptron_tree'),'model')
	
	tree = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False,balance_mode = balance_mode, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree.train(data_file = train_file, data_path = data_path, epochs_per_node = epochs_per_node,
	 batch_size = batch_size, model_save_path = model_save_path)
	tree.save(os.path.join(model_save_path, 'tree_model_Perceptron_treeWDB'))
	
	
	tree_test = DTree(num_classes = num_classes, num_child = num_child, max_depth = max_depth, data_type = data_type, 
		data_dimension = data_dimension, data_balance = False, balance_mode = balance_mode, decision_type = 'Perceptron', 
		count_threshold = count_threshold, purity_threshold = purity_threshold, impurity_drop_threshold = impurity_drop_threshold)
	tree_test.predict(model_save_file = os.path.join(model_save_path, 'tree_model_Perceptron_treeWDB'), 
		model_save_path = model_save_path, data_file= os.path.join(data_path, test_file), data_path = data_path,
		 output_file = output_file)

def decision_tree(data_path, train_file, test_file, output_file):
	os.path.join(data_path, train_file)
	dataset = pd.read_csv(os.path.join(data_path, train_file))
	features = [col for col in dataset.columns if col!='label']
	X = dataset.as_matrix(features)
	Y = dataset.as_matrix(['label'])
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X, Y)
	print('training done')
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

def svm(data_path, train_file, test_file, output_file):
	os.path.join(data_path, train_file)
	dataset = pd.read_csv(os.path.join(data_path, train_file))
	features = [col for col in dataset.columns if col!='label']
	X = dataset.as_matrix(features)
	Y = dataset.as_matrix(['label'])
	clf = SVC(verbose = True)
	clf = clf.fit(X, Y)
	print('training done')
	test = pd.read_csv(os.path.join(data_path, test_file))
	cols = [col for col in test.columns if col not in ['predicted_label', 'label', 'assigned_node']]
	
	x = test.as_matrix(cols)
	pred = clf.predict(x)
	#print(pred)

	test['predicted_label'] = pred
	test = test[['label','predicted_label']]
	test.to_csv(os.path.join(os.path.join(data_path, 'SVM'), output_file), index = False)
	acc = pd.np.sum(test['label']==test['predicted_label']).astype(pd.np.float32) / len(test)
	print("Accuracy: {}".format(acc))
	f = open(os.path.join(os.path.join(data_path, 'SVM'),'accuracy.txt'), 'w')
	f.write("Accuracy: {}".format(acc))
	f.close()

def mlp(data_path, train_file, test_file, output_file):
	os.path.join(data_path, train_file)
	dataset = pd.read_csv(os.path.join(data_path, train_file))
	features = [col for col in dataset.columns if col!='label']
	X = dataset.as_matrix(features)
	Y = dataset.as_matrix(['label'])
	mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

	mlp.fit(X, Y)
	print('training done')
	test = pd.read_csv(os.path.join(data_path, test_file))
	cols = [col for col in test.columns if col not in ['predicted_label', 'label', 'assigned_node']]
	
	x = test.as_matrix(cols)
	pred = mlp.predict(x)
	#print(pred)

	test['predicted_label'] = pred
	test = test[['label','predicted_label']]
	test.to_csv(os.path.join(os.path.join(data_path, 'SVM'), output_file), index = False)
	acc = pd.np.sum(test['label']==test['predicted_label']).astype(pd.np.float32) / len(test)
	print("Accuracy: {}".format(acc))
	f = open(os.path.join(os.path.join(data_path, 'MLP'),'accuracy.txt'), 'w')
	f.write("Accuracy: {}".format(acc))
	f.close()

def clean_dir_files(folder):
	if not os.path.isdir(folder):
		return
	for f in os.listdir(folder):
		ob = os.path.join(folder,f)
		if os.path.isdir(ob):
			clean_dir_files(ob)
		else:
			os.remove(ob)

if __name__ == "__main__":
	#d_sets = {'connect4':{'num_classes':3, 'data_dimension':126},'dna':{'num_classes':3, 'data_dimension':180},'letterdata':{'num_classes':26, 'data_dimension':16},'pendigits':{'num_classes':10, 'data_dimension':16},'protein':{'num_classes':3, 'data_dimension':357}}

	#d_sets = {'satimage':{'num_classes':6, 'data_dimension':36},'seismic':{'num_classes':3, 'data_dimension':50},'sensorless':{'num_classes':11, 'data_dimension':48},'shuttle':{'num_classes':7, 'data_dimension':9},'vowel':{'num_classes':11, 'data_dimension':10}}

	#d_sets = {'sensorless':{'num_classes':11, 'data_dimension':48}}

	d_sets = {'vowel':{'num_classes':11, 'data_dimension':10}}

	#d_sets = {'dna':{'num_classes':3, 'data_dimension':180}}

	#d_sets = {'connect4':{'num_classes':3, 'data_dimension':126}}
	#d_sets = {'letterdata':{'num_classes':26, 'data_dimension':16}}
	#d_sets = {'pendigits':{'num_classes':10, 'data_dimension':16}}
	#d_sets = {'protein':{'num_classes':3, 'data_dimension':357}}
	#d_sets = {'shuttle':{'num_classes':7, 'data_dimension':9}}


	for D_SET,QUAL in d_sets.items():
		#D_SET = 'letterdata'
		data_path = './datasets/'+D_SET
		train_file = D_SET+'_train.csv'
		test_file = D_SET+'_test.csv'
		output_file= 'prediction_'+D_SET+'.csv'
		num_classes = QUAL['num_classes']
		num_child = 2
		max_depth = 20
		data_type = 'float'
		data_dimension = QUAL['data_dimension']
		balance_mode = 'under_sample'
	
		count_threshold = 20
		purity_threshold = 0.95
		impurity_drop_threshold = 0.05

		epochs_per_node = 200
		batch_size = 200

		header = True #should headers be added to csv
		lab_beginning = True # do labels occur in the first col of csv
		normalize = False #should data be normalized
		class_format = True # set to true if classes are labeled 1 to n
		preprocess_needed = False #is preprocessing needed (Keep this true only for running the first time)

		if preprocess_needed:

			preprocess(header, os.path.join(data_path,train_file),False, data_dimension, lab_beginning, normalize, class_format, num_classes)
			preprocess(header, os.path.join(data_path,test_file),True, data_dimension, lab_beginning, normalize, class_format, num_classes)
	
		# if not os.path.isdir(os.path.join(data_path,'perceptron_tree_hybrid')):
		# 	os.makedirs(os.path.join(os.path.join(data_path,'perceptron_tree_hybrid')))

		# clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))
		# print('training perceptron tree with data balancing')
		# perceptron_tree_hybrid(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, 
	 #    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		# # #shutil.copytree(os.path.join(data_path,'perceptron_tree_hybrid'),os.path.join(data_path,'perceptron_tree_hybrid_2c_db'))

		# # #clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))
		# print('training perceptron tree without data balancing')
		# perceptron_tree_hybridWDB(data_path, train_file, test_file, 'prediction_wdb.csv', num_classes, num_child, max_depth, data_type, 
	 #    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		# # #shutil.copytree(os.path.join(data_path,'perceptron_tree_hybrid'),os.path.join(data_path,'perceptron_tree_hybrid_2c_wdb'))

		# #clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))

		if not os.path.isdir(os.path.join(data_path,'perceptron_tree_hybrid')):
			os.makedirs(os.path.join(os.path.join(data_path,'perceptron_tree_hybrid')))

		#clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))
		print('training perceptron tree with data balancing')
		perceptron_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, 
	    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size, balance_mode)
		#shutil.copytree(os.path.join(data_path,'perceptron_tree_hybrid'),os.path.join(data_path,'perceptron_tree_hybrid_2c_db'))

		#clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))
		print('training perceptron tree without data balancing')
		perceptron_treeWDB(data_path, train_file, test_file, 'prediction_wdb.csv', num_classes, num_child, max_depth, data_type, 
	    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size, balance_mode)
		#shutil.copytree(os.path.join(data_path,'perceptron_tree_hybrid'),os.path.join(data_path,'perceptron_tree_hybrid_2c_wdb'))

		#clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))

		# if not os.path.isdir(os.path.join(data_path,'perceptron_tree')):
		# 	os.makedirs(os.path.join(os.path.join(data_path,'perceptron_tree')))

		# clean_dir_files(os.path.join(data_path,'perceptron_tree'))
		# print('training perceptron tree with data balancing')
		# perceptron_tree(data_path, train_file, test_file, output_file, num_classes, num_child, max_depth, data_type, 
	 #    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		# #shutil.copytree(os.path.join(data_path,'perceptron_tree_hybrid'),os.path.join(data_path,'perceptron_tree_hybrid_2c_db'))

		# #clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))
		# print('training perceptron tree without data balancing')
		# perceptron_treeWDB(data_path, train_file, test_file, 'prediction_wdb.csv', num_classes, num_child, max_depth, data_type, 
	 #    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		#shutil.copytree(os.path.join(data_path,'perceptron_tree_hybrid'),os.path.join(data_path,'perceptron_tree_hybrid_2c_wdb'))

		#clean_dir_files(os.path.join(data_path,'perceptron_tree_hybrid'))

		print('{} complete'.format(D_SET))
		print('+++++++++++')
		# print('training perceptron tree with 3 child nodes')
		# perceptron_treeWDB(data_path, train_file, test_file, 'prediction_3child_wdb.csv', num_classes, 3, max_depth, data_type, 
	 #    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		# perceptron_tree(data_path, train_file, test_file, 'prediction_3child.csv', num_classes, 3, max_depth, data_type, 
	 #    	data_dimension, count_threshold, purity_threshold, impurity_drop_threshold, epochs_per_node, batch_size)
		# print('Training mlp')
		# mlp(data_path, train_file, test_file, output_file)
		# print('Training decision tree')
		# decision_tree(data_path, train_file, test_file, output_file)
		# print('Training svm')
		# svm(data_path, train_file, test_file, output_file)
	

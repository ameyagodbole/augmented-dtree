# augmented-dtree
Implementation of **Progressively Balanced Multi-class Neural Trees**
### Link to paper
https://ieeexplore.ieee.org/abstract/document/8599945

# Model Discription and implementation details

## Augmented DTree
Augmented dtree is a decision tree implementation where each node of the decision tree is itself a classifier of class DTNode. These nodes are processed in a queue and new nodes are added to the end of the queue to be processed. Each node can have 2 or more child nodes. To keep track of data going into each child node, a seperate file is created for the data going into each child node from the parent node. Also, when data balancing is called a seperate file is created for the data balanced data at each node.

## DTNode
DTNode comprises of the individual nodes that the tree is made of. Each node of the tree has a classifier which is passed as an argument. The node learns the classifier parameters using the data received by the node and calculates the entropy at the node.

## Databalance
It takes as parameters the type of databalancing to be performed and the file whose data is to be balanced. It then performs databalancing and saves the balanced data in a new file.

# Usage

Training the model:

1. Create an instance of the class DTree
	Parameters:

	num_classes: Number of classes in the dataset
	num_clild: Number of child nodes required for each node
	max_depth: Maximum depth of the neural tree allowed (to preven overfitting)
	data_type: Data type of the data 
	data_dimension: Number of features in the data
	data_balance: Whether data balancing is required (True/False)
	balance_mode: Type of data balancing required
	decision_type: The typw of classifier to be used in each node of the tree
	decision_criterion: The criterion used to decide when a nore is a leaf node
	purity_threshold: The maximum purity of a node above which is is declared a leaf node
	count_threshold: The minimum count of data points at a node, below which we declare the node as a leaf node to prevent overfitting
	impurity_drop_threshold: The minimum impurity drop from parent to child required, below which we declare the node as a leaf node

	For example:

		 tree = DTree(num_classes = 3, num_child = 2, max_depth = 5, data_type = 'float', data_dimension = 4,
	 			data_balance = True, balance_mode='kmeans_os_no_us', decision_type = 'C45', decision_criterion='entropy',
	 			purity_threshold = 0.7, count_threshold = 5, impurity_drop_threshold = 0.1)

2. Train the tree
	Parameters:

	data_file: the file containing the dataset
	working_dir_path: the file we store the output in 
	epochs_per_node: number of epochs
	batch_size: batch size for training
	model_save_path: path to save the model in

	For example:

		tree.train(data_file = './output/iris_train.csv', working_dir_path='./output/c45_iris/',epochs_per_node = 200, batch_size = 20, model_save_path = None)

3. Save the trained model
	Parameters:

	model_save_file: name of the model including path

	For example:
		tree.save(path.join('./output/C45_iris/', 'test_C45_iris.pkl'))

Using a saved model for testing:

1. Create an instance of class DTree with same parameters used for training

	For example:
	
		tree_test = DTree(num_classes = 3, num_child = 2, max_depth = 5, data_type = 'float', 
					data_dimension = 4, data_balance = True, decision_type = 'C45', count_threshold = 5,
					purity_threshold = 0.7, impurity_drop_threshold = 0.1)

2. Predict the labels using a saved model
	Parameters:

	model_save_file: name of the model file as saved
	model_save_path: name of the path where the model is saved
	data_file: the test dataset
	working_dir_path: Path of the working directory
	output_file: name of the file to store the output

	For example

		tree_test.predict(model_save_file = path.join('./output/c45_iris/', 'test_c45_iris.pkl'), 
	 		model_save_path = path.join('./output/c45_iris/', 'model'), data_file= './output/iris_test.csv',
	 		working_dir_path = './output/c45_iris/', output_file = 'c45_predict.csv')

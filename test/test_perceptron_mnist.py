import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from augmented_dtree import DTree

def main():
	tree = DTree(num_classes = 10, num_child = 2, max_depth = 20, data_type = 'float', 
		data_dimension = 784, data_balance = False, decision_type = 'Perceptron', count_threshold = 10,
		 purity_threshold = 0.6, impurity_drop_threshold = 0.1)
	tree.train(data_file = './output/perceptron_mnist/mnist_train1.csv', epochs_per_node = 50, batch_size = 200, model_save_path = './output/perceptron_mnist/model/')
	tree.save('./output/perceptron_mnist/tree_model')

	tree_test = DTree(num_classes = 10, num_child = 2, max_depth = 20, data_type = 'float', 
		data_dimension = 784, data_balance = False, decision_type = 'Perceptron', count_threshold = 10,
		 purity_threshold = 0.6, impurity_drop_threshold = 0.1)
	tree_test.predict(model_save_file='./output/perceptron_mnist/tree_model',
	 model_save_path = './output/perceptron_mnist//model/', data_file= './output/perceptron_mnist/mnist_test1.csv')


if __name__ == "__main__":
    main()
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from augmented_dtree import DTree

def main():
	tree = DTree(num_classes = 3, num_child = 2, max_depth = 5, data_type = 'float', data_dimension = 4,
	 data_balance = True, balance_mode='kmeans_os_no_us', decision_type = 'Perceptron', decision_criterion='entropy',
	 purity_threshold = 0.7, count_threshold = 5, impurity_drop_threshold = 0.1)
	tree.train(data_file = './output/iris_train.csv', working_dir_path='./output/perceptron_iris/',
	 epochs_per_node = 200, batch_size = 20, model_save_path = None)
	tree.save(path.join('./output/perceptron_iris/', 'test_perceptron_iris.pkl'))

	tree_test = DTree(num_classes = 3, num_child = 2, max_depth = 5, data_type = 'float', data_dimension = 4,
	 data_balance = True, balance_mode='kmeans_os_no_us', decision_type = 'Perceptron', decision_criterion='entropy',
	 purity_threshold = 0.7, count_threshold = 5, impurity_drop_threshold = 0.1)
	tree_test.predict(model_save_file = path.join('./output/perceptron_iris/', 'test_perceptron_iris.pkl'), 
	 model_save_path = path.join('./output/perceptron_iris/', 'model'), data_file= './output/iris_test.csv',
	 working_dir_path = './output/perceptron_iris/', output_file = 'perceptron_predict.csv')

if __name__ == "__main__":
    main()
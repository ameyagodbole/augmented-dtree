import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from augmented_dtree import DTree

def main():
	tree = DTree(num_classes = 3, num_child = 2, max_depth = 5, data_type = 'float', 
		data_dimension = 4, data_balance = True, decision_type = 'Perceptron', count_threshold = 5,
		 purity_threshold = 0.7, impurity_drop_threshold = 0.1)
	tree.train(data_file = './output/perceptron_iris/iris_train.csv', epochs_per_node = 200, batch_size = 20, model_save_path = './output/perceptron_iris/model/')


if __name__ == "__main__":
    main()
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from augmented_dtree import DTree

def main():
	tree = DTree(num_classes = 3, num_child = 2, max_depth = 5, data_type = 'float', 
		data_dimension = 4, data_balance = True, decision_type = 'C45', count_threshold = 5,
		 purity_threshold = 0.7, impurity_drop_threshold = 0.1)
	tree.train(data_file = './output/iris_train.csv', working_dir_path='./output/C45_iris/', epochs_per_node = 200, batch_size = 20, model_save_path = None)
	tree.save(path.join('./output/C45_iris/', 'test_C45_iris.pkl'))
	
	tree_test = DTree(num_classes = 3, num_child = 2, max_depth = 5, data_type = 'float', 
		data_dimension = 4, data_balance = True, decision_type = 'C45', count_threshold = 5,
		 purity_threshold = 0.7, impurity_drop_threshold = 0.1)
	tree_test.predict(model_save_file = path.join('./output/C45_iris/', 'test_C45_iris.pkl'), 
		model_save_path = path.join('./output/C45_iris/', 'model'), data_file= './output/iris_test.csv', working_dir_path = './output/C45_iris/',
		 output_file = 'C45_predict.csv')

if __name__ == "__main__":
    main()
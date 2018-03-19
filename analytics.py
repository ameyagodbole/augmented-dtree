import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import os

# TODO: Possible additional features
# 1. More options/details in draw_tree
# 2. Analyse ratio based impurity threshold
# 3. Analyse stopping criteria

def draw_tree(model_file, train_file, data_path, graph_file, img_file=None, label_colors=None, legend=False):
	"""
	Arguments:
	---
	model_file : Saved DTree model
	train_file : Data used for training the tree
	data_path : Folder containing the splits generated during the training of the tree
			   i.e. data_*.csv
	graph_file : The DOT file for saving tree representation
	img_file : The PNG file for saving tree image (optional)
	label_colors : List of colors to assign to the labels
	legend : Display legend with the graph
	"""
	with open(model_file,'r') as f:
		tree = pickle.load(f)

	nodes = ''
	connections = ''

	df = pd.read_csv(os.path.join(train_file))
	num_classes = len(np.unique(df['label']))

	if not label_colors:
		HSV_tuples = [(x*1.0/(num_classes), 0.5, 1.0) for x in range(num_classes)]
		label_colors = [clr.to_hex(clr.hsv_to_rgb(x)) for x in HSV_tuples]
	else:
		if len(label_colors)!=num_classes:
			raise ValueError('len(label_colors)!=num_classes')

	ranks = {}

	for i in range(len(tree.keys())):
		try:
			df = pd.read_csv(os.path.join(data_path,'data_{}.csv'.format(i)))
		except IOError:
			if i==0:
				df = pd.read_csv(train_file)
			else:
				raise IOError('Could not find file {}')
		
		out = '\t' + str(i) + ' [label="{<f1>' + str(i) + '|{'
		for c in range(num_classes):
			if c == (num_classes/2):
				out += '<f0>'
			out += str(len(df.loc[df['label']==c]))
			if c<(num_classes-1):
				out	+= '|'
			else:
				out += '}}"'

		if tree[i]['is_decision_node']:
			out += ', style=filled, fillcolor="{}"];\n'.format(label_colors[tree[i]['label']])
		else:
			out += '];\n'
		
		nodes += out

		try:
			ranks[tree[i]['node_depth']] += (str(i)+'; ')
		except KeyError:
			ranks[tree[i]['node_depth']] = (str(i)+'; ')

		if tree[i]['num_child'] == 0:
			continue

		out = '\t' + str(i) + ' -> {'
		for c in tree[i]['child_id']:
			out += str(c) + ', '
		out = out[:-2]
		out += '};\n'

		connections += out

	graph = 'digraph G {\n\tnode [shape=record];\n\n'
		
	graph += nodes
	graph += '\n'
	graph += connections
	graph += '\n'

	for r in ranks.keys():
		graph += '\t{rank = same; ' + ranks[r] +'}\n'

	if legend:
		graph += '\n'
		graph += '\tlegend [shape=plaintext, label=<<table border="0" cellspacing="0"><tr>'
		for c in range(num_classes):
			graph += '<td border="1">{}</td>'.format(c)
		graph += '</tr><tr>'
		for c in range(num_classes):
			graph += '<td border="1" bgcolor="{}"> </td>'.format(label_colors[c])
		graph += '</tr></table>>];\n\n'

	graph += '}'

	with open(graph_file,'w') as f:
		f.write(graph)

	if img_file:
		cmd = 'dot -Tpng {} -o {}'.format(graph_file, img_file)
		os.system(cmd)


def analyse_perceptron_evaluations(datasets_dict, eval_save_file):
	"""
	Arguments:
	---
	datasets_dict : Dictionary with dataset name as key and value of a dictionary with corresponding prediction file
				   e.g. dsets_dict = {'dset0':{'prediction_file':'./f0/file0.csv'},'dset1':{'prediction_file':'./f1/file1.csv'}}
	eval_save_file : CSV file to save evaluations
	"""
	percep_eval_df = pd.DataFrame(columns=['dataset','avg','max','min','mlp'])
	for dset_name,dset in datasets_dict.iteritems():
		df = pd.read_csv(dset['prediction_file'])
		avg_perceptron_eval = 2*np.mean(df['label_depth'])
		max_perceptron_eval = 2*np.max(df['label_depth'])
		min_perceptron_eval = 2*np.min(df['label_depth'])
		tmp_df = pd.DataFrame(data={'dataset':dset_name, 'avg':avg_perceptron_eval,'max':max_perceptron_eval,
									'min':min_perceptron_eval,'mlp':k*(dset['num_classes']+dset['data_dimension'])},
							  index=[0])
		percep_eval_df = pd.concat([percep_eval_df,tmp_df],ignore_index=True)
	percep_eval_df[['dataset','avg','max','min','mlp']]

	percep_eval_df.to_csv(eval_save_file, index=False)


def impurity_vs_depth(model_file, train_file, data_path, dset, img_file=None, display_graph=True, verbose=False):
	'''
	Arguments:
	---
	model_file : Saved DTree model
	train_file : File containing the full training data
	data_path : Folder containing the data splits created during training
	dset : Graph title
	img_file : File to save graph
	display_graph : Whether to display the graph
	verbose : Print intermediate output
	'''
	with open(model_file, 'r') as pkl_file:
		tree = pickle.load(pkl_file)
	
	full_data = pd.read_csv(train_file)
	
	def get_impurity(df):
		impurity_score = 0.0
		for c in np.unique(df['label']):
			p = float(len(df.loc[df['label']==c]))/len(df)
			impurity_score -= p*np.log2(p)
		return impurity_score
	
	max_depth = 0
	for _,n in tree.iteritems():
		max_depth = max(max_depth, n['node_depth'])
	root = tree[0]
	impurity = np.zeros(max_depth+1)
	depth = np.array(range(max_depth+1))
	points_at_depth = np.zeros(max_depth+1)
	
	for nid,n in tree.iteritems():
		if nid != 0:
			df = pd.read_csv(os.path.join(data_path,'data_{}.csv'.format(nid)))
		else:
			df = pd.read_csv(train_file)
		impurity[n['node_depth']] += get_impurity(df)*len(df)
		points_at_depth[n['node_depth']] += len(df)

	if np.any(points_at_depth != points_at_depth[0]):
		print points_at_depth
		print 'Missing points. Check input file.'
		return

	plt.plot(depth, impurity/points_at_depth, label=label)
	plt.title(dset)
	plt.xlabel('depth')
	plt.ylabel('weighted average impurity')
	plt.xticks(range(len(depth)))

	if display_graph:
		plt.show()
	if img_file:
		plt.savefig(img_file)


def label_ratios(model_file, train_file, img_file=None, display_graph=True, verbose=False):
	'''
	Arguments:
	---
	model_file : Saved DTree model
	train_file : Data used for training the tree
	img_file : File to save graph
	display_graph : Whether to display the graph
	verbose : Print intermediate output
	'''
	df = pd.read_csv(train_file)
	data_label_counts = dict(Counter(df['label']))
	del df
	key = data_label_counts.keys()
	data_label_counts = np.asarray([data_label_counts[k] for k in key], dtype=np.float32)
	data_label_counts /= np.sum(data_label_counts)
	
	with open(model_file, 'r') as pkl_file:
		tree = pickle.load(pkl_file)
	tree_label_counts = {}
	for i in tree.keys():
		if tree[i]['is_decision_node']:
			try:
				tree_label_counts[tree[i]['label']] += 1
			except KeyError:
				tree_label_counts[tree[i]['label']] = 1
	del tree
	tree_label_counts = Counter(tree_label_counts)
	tree_label_counts = np.asarray([tree_label_counts[k] for k in key], dtype=np.float32)
	tree_label_counts /= np.sum(tree_label_counts)
	
	if verbose:
		print 'train_label_ratio: ',tree_label_counts
		print 'data_label_ratio: ',data_label_counts

	plt.plot(key, tree_label_counts, label='tree_label_count')
	plt.plot(key, data_label_counts, label='data_label_count')
	plt.legend()
	plt.grid(True, axis='y')
	if display_graph:
		plt.show()
	if img_file:
		plt.savefig(img_file)


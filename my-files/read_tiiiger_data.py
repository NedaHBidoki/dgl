import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from network_analytics import *

def load_npz_data():
	data = np.load("reddit.npz")
	#print('adj',adj)
	print(type(data))
	#print(data.files)
	print('y_train',data['y_train'].shape)
	print('y_val',data['y_val'].shape)
	print('y_test',data['y_test'].shape)
	print('train_index',data['train_index'].shape)
	print('val_index',data['val_index'].shape)
	print('test_index',data['test_index'].shape)
	#print('adj',adj.shape)
	print('feats',data['feats'].shape)
	print('1 in y_test:', sum(data['y_test']))

def load_pkl_data(file_):
	with open(file_, 'rb') as f:
		if sys.version_info > (3, 0):
			data =pkl.load(f, encoding='latin1')
		else:
			data = pkl.load(f)
	return data

def create_net(graph):
	#adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
	#G=nx.from_numpy_matrix(adj)
	#G = nx.from_scipy_sparse_matrix(adj)
	G = nx.from_dict_of_lists(graph)
	nx.draw(G)
	plt.savefig('before.png')
	plt.close()
	G = remove_isolated_nodes(G)
	nx.draw(G)
	plt.savefig('after.png')
	return G

def generate_A_attrM(dataset,attr_file):
	graph = load_pkl_data(graph_file)
	G = create_net(graph)
	#G = get_G(dataset)
	#A = 
	attrM = get_attrM(attr_file)
	return A,attrM

def get_intersect(G,A):
	A_pd = A_pd[A_pd.index.isin(kibana_matrix.index)]
	A_pd = A_pd[np.intersect1d(A_pd.columns, kibana_matrix.index)]
	kibana_matrix = kibana_matrix[kibana_matrix.index.isin(A_pd.index)]
	print('A_pd.shape', A_pd.shape)
	print('kibana_matrix.shape', kibana_matrix.shape)
	return A_pd, kibana_matrix

def get_zachary_graph():
	db='karate_club'
	base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/process_data/karate_club_data/'
	wiki_file = 'adjacency.csv'
	df = pd.read_csv(base + wiki_file, sep=' ')
	nodes = df['nodeOut'].values.tolist()
	parents = df['nodeIn'].values.tolist()
	all_nodes = set(nodes + parents)
	edges = zip(nodes,parents)
	G = nx.Graph()
	G.add_edges_from(list(edges))
	#A = nx.adjacency_matrix(G)
	return G

def main():
	dataset_str="citeseer"
	plots_dir = 'plots/%s/'%dataset_str
	if not os.path.exists(plots_dir):
		os.makedirs(plots_dir)
	graph_file = "data/ind.{}.{}".format(dataset_str.lower(), 'graph')
	attr_file = "data/ind.{}.{}".format(dataset_str.lower(), 'allx')
	graph = load_pkl_data(graph_file)
	attr = load_pkl_data(attr_file)
	print(attr)
	sys.exit()
	G = create_net(graph)
	plot_network_stat(G,plots_dir)
	
	percents = [1,2,3]
	for p in percents:
		dir_ = 'plots/%s/rm_%s/'%(dataset_str,p)
		if not os.path.exists(dir_):
			os.makedirs(dir_)
		nodes = get_top_betweenness(G,p)

def test():
	dataset_str="Zachary"
	plots_dir = 'plots/%s/'%dataset_str
	if not os.path.exists(plots_dir):
		os.makedirs(plots_dir)
	G = get_zachary_graph()
	plot_network_stat(G,plots_dir)
	percents = [1,2,3]
	print('-----Network Size:',len(G.nodes()))
	for p in percents:
		dir_ = 'plots/%s/rm_%s/'%(dataset_str,p)
		if not os.path.exists(dir_):
			os.makedirs(dir_)
		nodes = get_top_betweenness(G,p)
		G = remove_nodes(G, nodes)
		print('-----Network Size After Removeing %s Percent (%s Nodes):'%(p,len(nodes)),len(G.nodes()))
		plot_network_stat(G,dir_)
main()
	

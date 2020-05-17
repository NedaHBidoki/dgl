import itertools
import numpy as np
import networkx as nx
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import argparse

parser = argparse.ArgumentParser(description='SGC')
register_data_args(parser)
parser.add_argument("--gpu", type=int, default=-1,
    help="gpu")
parser.add_argument("--lr", type=float, default=0.2,
    help="learning rate")
parser.add_argument("--weight-decay", type=float, default=5e-6,
    help="Weight for L2 loss")
parser.add_argument("--dir_", type=str, default='test_plots/',
    help="dir_ e.g. test_plots/ ")
parser.add_argument("--method", type=str, default='A_test',
    help="dir_ e.g. A^2+I/ ")

args = parser.parse_args()
print(args)

def random_walk(G, n_samples=5, n_nodes_per_sample = 5):
	n_nodes = len(G.nodes())
	n_edges = len(G.edges())

	# avg_k (population): 859.75
	avg_k = np.mean(
	    [ float(n[1]) / 2 for n in G.degree() ])

	print('here 1')
	def random_walk_generator(G):
		v = np.random.choice(list(G.nodes()))
		for _ in itertools.count():
			v = np.random.choice(list(G.neighbors(v)))
			yield v


	print('here 2')
	generators = [ 
	    random_walk_generator(G) for sample in range(0, n_samples) ]

	random_walk_samples = [
	    [ generator.__next__() \
		for step in range(0, n_nodes_per_sample) ] \
	   for generator in generators 
	]
	
	return [j for i in random_walk_samples for j in i]
def sample_data(gl,features, labels, sampled_nodes):
		#print(gl.nodes().tolist())
		nodes = gl.nodes().tolist()
		nodes_to_remove = [n for n in nodes if n not in sampled_nodes]
		gl.remove_nodes(nodes_to_remove)
		features = np.delete(features, nodes_to_remove, 0)
		labels = np.delete(labels, nodes_to_remove, 0)
		return gl, features, labels

def main():
	n_users = 100
	n_relationships = 10

	data = load_data(args)
	gl = DGLGraph(data.graph)
	features, labels = data.features, data.labels
	G = gl.to_networkx()
	#print(gl.nodes().tolist())
	features, labels = data.features, data.labels
	sampled_nodes = random_walk(G)
	print(' features:',len(features))
	print(' labels:',len(labels))
	print(' nodes:',len(gl.nodes()))
	gl, features, labels = sample_data(gl,features, labels, sampled_nodes)
	print(' features:',len(features))
	print(' labels:',len(labels))
	print(' nodes:',len(gl.nodes()))

main()

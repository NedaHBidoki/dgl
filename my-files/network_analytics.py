import pandas as pd
import sys
import random
import networkx as nx
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.sparse
import numpy
import os
from networkx.algorithms.community import greedy_modularity_communities
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from sklearn.model_selection import train_test_split
import math
import numpy as np
from scipy.sparse import coo_matrix
import csv
import random
from numpy.linalg import matrix_power
def network_preprocess(dgl_g, features, labels, dir_, dataset, percent, criteria, fold):
	prefix = 'weakties_%s_%sp_%s_%s'%(dataset, percent, criteria,fold)
	node_list = dgl_g.nodes().tolist()
	G = dgl_g.to_networkx()

	########## this funstion is added  to create theh weak ties  network ########
	#G = get_weak_ties_network(G,node_list)


	#print('Network stat BEFORE removing targeted nodes ....')
	#print(nx.info(G))

	temp_G = G.to_undirected()
	top_nodes = get_top_nodes(temp_G, percent, criteria)
	G = remove_top_nodes(G, top_nodes)
	#print('Network stat AFTER removing targeted nodes ....')
	#print(nx.info(G))

	plot_network_stat(G.to_undirected(),dir_, prefix)

	remaining_nodes = [x for x in G.nodes if x not in top_nodes]
	features,labels, train_mask,val_mask,test_mask = get_model_parameters(remaining_nodes, features, labels)

	node_list = [n for n  in node_list if n not in top_nodes]
	g = reconstruct_g(G,node_list)

	
	return g,features,labels,train_mask,val_mask,test_mask

def get_weak_ties_network(G,node_list):
	A = nx.to_numpy_matrix(G, nodelist = node_list)
	A2 = matrix_power(A,2)
	A2 = A2 - A
	print(sum(np.diagonal(A2)))
	G = nx.from_numpy_matrix(A2)
	return G

def get_model_parameters(top_nodes, features, labels):
	#print('top_nodes:',len(top_nodes))
	x_indices = range(len(features))
	##print('x_indices:',x_indices)
	msk = [(el in top_nodes) for el in x_indices]
	#print('len msk:',sum(msk))
	#print('features before mask:',len(features))
	features = features[msk]
	#print('features after mask:',len(features))
	#print('labels before mask:',len(labels))
	labels = labels[msk]
	#print('labels after mask:',len(labels))
	x_indices = np.asarray(x_indices, dtype=np.float32)
	#print('x_indices before mask:',len(x_indices))
	x_indices = x_indices[msk]
	#print('x_indices after mask:',len(x_indices))
	#print('labels:',labels)
	x_indices = range(len(features)) ####?????????????????
	x_train, x_test, y_train, y_test = train_test_split(x_indices, x_indices, test_size=0.3)
	x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
	all_indices = range(len(features))
	train_mask = [(el in x_train) for el in all_indices]
	val_mask = [(el in x_val) for el in all_indices]
	test_mask = [(el in x_test) for el in all_indices]
	#print('train_mask:',sum(train_mask))
	#print('val_mask:',sum(val_mask))
	#print('test_mask:',sum(test_mask))
	#print('train_mask:',len(x_train))
	#print('val_mask:',len(x_val))
	#print('test_mask:',len(x_test))
	return features,labels, train_mask,val_mask,test_mask 

def reconstruct_g(G, node_list):
	A = nx.to_numpy_matrix(G, nodelist = node_list)
	sA = scipy.sparse.csr_matrix(A)
	g = DGLGraph()
	g.from_scipy_sparse_matrix(sA)
	return g

def remove_isolated_nodes(G):
	iso_nodes = list(nx.isolates(G))
	G.remove_nodes_from(iso_nodes)
	return G

def remove_top_nodes(G, nodes):
	G.remove_nodes_from(nodes)
	return G

def plot_network_stat(G,dir_, prefix):
	#plot_comps_size(G,dir_,prefix)
	plot_comps_rank(G,dir_,prefix)
	plot_nodes_degree(G,dir_,prefix)
	#plot_coms_seperately(G, dir_)
	write_network_additional_stat(G,dir_,prefix)
	'''
	try:
		plot_communities_stat(G,dir_,prefix)
	except:
		print('ERROR IN COMMUNITY DETECTION')
	'''
def plot_comps_size(G,dir_,prefix):
	comps = list(nx.connected_components(G))
	comps.sort(key=len, reverse= True)
	clens = [len(c) for c in comps]
	_ = plt.hist(clens, bins='auto',color='gray')
	plt.ylabel('Size')
	plt.xlabel('Frequency')	
	plt.title("Components size")
	plt.grid()
	plt.savefig(dir_+'plots/%s_comp_sizes.pdf'%prefix)
	plt.close()

	

def plot_comps_rank(G,dir_,prefix):
	comps = list(nx.connected_components(G))
	comps.sort(key=len, reverse= True)
	comp_sequence = sorted([len(c) for c in comps], reverse=True)
	# print "component sequence", component_sequence
	'''
	plt.loglog(comp_sequence, 'b-', marker='o')
	plt.title("Component rank plot")
	plt.ylabel("component size")
	plt.xlabel("rank")
	plt.grid()
	plt.savefig(dir_+'plots/%s_component_rank_connected_loglog.pdf'%prefix)
	plt.close()
	'''
	csvfile = dir_+'files/%s_component_rank_connected.csv'%prefix
	with open(csvfile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		for val in comp_sequence:
			writer.writerow([val]) 

def plot_communities_stat(G,dir_,prefix):
	# plot communties stat
	Communities = list(greedy_modularity_communities(G))
	clens = [len(c) for c in Communities]
	#print('communities len:', clens)
	'''
	_ = plt.hist(clens, bins='auto',color='blue')
	plt.ylabel('Size')
	plt.xlabel('Frequency')	
	plt.title("Community sizes")
	plt.grid()
	plt.savefig(dir_+'plots/%s_communities_sizes.pdf'%prefix)
	plt.close()
	'''
	csvfile = dir_+'files/%s_communities_sizes.csv'%prefix
	with open(csvfile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		for val in clens:
			writer.writerow([val]) 

def plot_nodes_degree(G,dir_, prefix):
	##print(nx.diameter(G)) #answer INF
	degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
	dmax = max(degree_sequence)
	#print(dmax)
	'''
	plt.loglog(degree_sequence, 'b-', marker='o')
	plt.title("Degree rank plot")
	plt.ylabel("degree")
	plt.xlabel("rank")
	plt.grid()
	plt.savefig(dir_+'plots/%s_nodes_degrees.pdf'%prefix)
	plt.close()
	'''
	csvfile = dir_+'files/%s_nodes_degrees.csv'%prefix
	with open(csvfile, "w") as output:
		writer = csv.writer(output, lineterminator='\n')
		for val in degree_sequence:
			writer.writerow([val]) 

def plot_communities_seperately(G, dir_):
	Communities = list(greedy_modularity_communities(G))	
	C = [G.subgraph(c).copy() for c in Communities]
	for i,c in enumerate(C):
		nx.draw(c)
		plt.savefig(dir_+ 'plots/%s.png'%i)

def write_network_additional_stat(G,dir_,prefix):
	info = {}
	#print(nx.info(G))
	density = nx.density(G)
	#print("Network density:", density)
	info['density'] = density
	components = nx.connected_components(G)
	largest_component = max(components, key=len)
	# Create a "subgraph" of just the largest component
	# Then calculate the diameter of the subgraph, just like you did with density.
	subgraph = G.subgraph(largest_component)
	diameter = nx.diameter(subgraph)
	#print("Network diameter of largest component:", diameter)
	info['lc_diameter'] = diameter
	triadic_closure = nx.transitivity(G)
	#print("Triadic closure:", triadic_closure)
	info['triadic_closure'] = triadic_closure
	with open(dir_ + 'files/net_info/%s_info.csv'%prefix, 'w') as f:
		for key in info.keys():
			f.write("%s,%s\n"%(key,info[key]))

def get_top_betweenness(G,percent):
	numOfnodes = math.ceil(int(percent) * len(G.nodes())/100)
	degree_dict = dict(G.degree(G.nodes()))
	nx.set_node_attributes(G, degree_dict, 'degree')
	betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality
	#eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality

	# Assign each to an attribute in your network
	nx.set_node_attributes(G, betweenness_dict, 'betweenness')
	#nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
	sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)

	#print("Top %s nodes by betweenness centrality:"%numOfnodes)
	#for b in sorted_betweenness[:numOfnodes]:
	    #print(b)
	#First get the top 20 nodes by betweenness as a list
	top_betweenness = sorted_betweenness[:numOfnodes]

	#Then find and print their degree
	for tb in top_betweenness: # Loop through top_betweenness
	    degree = degree_dict[tb[0]] # Use degree_dict to access a node's degree, see footnote 2
	    #print("Name:", tb[0], "| Betweenness Centrality:", tb[1], "| Degree:", degree)
	return [tb[0] for tb in top_betweenness]

def get_top_closeness(G,percent):
	numOfnodes = math.ceil(int(percent) * len(G.nodes())/100)
	degree_dict = dict(G.degree(G.nodes()))
	nx.set_node_attributes(G, degree_dict, 'degree')
	closeness_dict = nx.closeness_centrality(G) # Run betweenness centrality


	# Assign each to an attribute in your network
	nx.set_node_attributes(G, closeness_dict, 'closeness')
	sorted_closeness = sorted(closeness_dict.items(), key=itemgetter(1), reverse=True)

	#print("Top %s nodes by closeness centrality:"%numOfnodes)
	#for b in sorted_closeness[:numOfnodes]:
	    #print(b)
	#First get the top 20 nodes by betweenness as a list
	top_closeness = sorted_closeness[:numOfnodes]

	#Then find and print their degree
	for tc in top_closeness: # Loop through top_betweenness
	    degree = degree_dict[tc[0]] # Use degree_dict to access a node's degree, see footnote 2
	    #print("Name:", tc[0], "| Closeness Centrality:", tc[1], "| Degree:", degree)
	return [tc[0] for tc in top_closeness]

def get_top_katzcentrality(G,percent):
	numOfnodes = math.ceil(int(percent) * len(G.nodes())/100)
	degree_dict = dict(G.degree(G.nodes()))
	nx.set_node_attributes(G, degree_dict, 'degree')
	katzcentrality_dict = nx.katz_centrality(G) # Run betweenness centrality


	# Assign each to an attribute in your network
	nx.set_node_attributes(G, katzcentrality_dict, 'katzcentrality')
	sorted_katzcentrality = sorted(katzcentrality_dict.items(), key=itemgetter(1), reverse=True)

	#print("Top %s nodes by katz centrality:"%numOfnodes)
	#for b in sorted_katzcentrality[:numOfnodes]:
	    #print(b)
	#First get the top 20 nodes by katzcentrality as a list
	top_katzcentrality = sorted_katzcentrality[:numOfnodes]

	#Then find and print their degree
	#for tb in top_katzcentrality: # Loop through top_betweenness
	    #degree = degree_dict[tb[0]] # Use degree_dict to access a node's degree, see footnote 2
	    #print("Name:", tb[0], "| katz Centrality:", tb[1], "| Degree:", degree)
	return [tb[0] for tb in top_katzcentrality]


def get_top_voterank(G, percent):
	numOfnodes = math.ceil(int(percent) * len(G.nodes())/100)
	#degree_dict = dict(G.degree(G.nodes()))
	#nx.set_node_attributes(G, degree_dict, 'degree')


	sorted_voterank = nx.voterank(G,numOfnodes) # Run voterank
	#sorted_voterank = sorted(voterank_dict.items(), key=itemgetter(1), reverse=True)

	#print("Top %s nodes by voterank:"%numOfnodes)
	#for b in sorted_voterank[:numOfnodes]:
	#    print(b)
	#First get the top 20 nodes by betweenness as a list
	#top_voterank = sorted_voterank[:numOfnodes]
	#Then find and print their degree
	#for tb in voterank_dict: # Loop through top_betweenness
	#   degree = degree_dict[tb[0]] # Use degree_dict to access a node's degree, see footnote 2
	#  print("Name:", tb[0], "| voterank:", tb[1], "| Degree:", degree)
	return sorted_voterank #[tb[0] for tb in top_voterank]

def get_top_random(G, percent):
	numOfnodes = math.ceil(int(percent) * len(G.nodes())/100)
	nodes = random.sample(G.nodes(),numOfnodes)
	return nodes

def remove_nodes(G, nodes):
	G.remove_nodes_from(nodes)
	return G

def write_communities(C,dir_):
	for i,c in enumerate(C):
		nx.write_edgelist(c, dir_ + "communities/community_%s_size%s.csv"%(i,len(c)), delimiter=',')
		df = pd.read_csv( dir_ + "communities/community_%s_size%s.csv"%(i,len(c)), names=['node','parent','w'])
		df['created_at']='nan'
		df[['created_at','node','parent']].to_csv( base + "communities/community_%s_size%s.csv"%(i,len(c)),index=False)

def write_components(C,dir_):
	# save components
	comps = list(nx.connected_components(G))
	comps.sort(key=len, reverse= True)
	S = [G.subgraph(c).copy() for c in comps]
	for i,g in enumerate(S[0:5]):
		nx.write_edgelist(G, "component_%s.csv"%i, delimiter=',')
		df = pd.read_csv("component_%s.csv"%i, names=['node','parent','w'])
		df['created_at']='nan'
		df[['created_at','node','parent']].to_csv("component_%s.csv"%i,index=False)
	# save components in a csv file
	comp_df = pd.DataFrame()
	comp_df['comp']= comps
	comp_df['comp #']=range(len(comps))
	comp_df['len'] = comp_df['comp'].apply(lambda x: len(x))
	comp_df.head().to_csv(base + '/connected_components.csv')
	#print(comp_df['len'].head())
	
def G_stat(G):	
	comps = list(nx.connected_components(G))
	comps.sort(key=len, reverse= True)
	S = [G.subgraph(c).copy() for c in comps]	
	largest_cc = max(comps, key=len)
	
	# draw graph in inset
	plt.axes([0.45, 0.45, 0.45, 0.45])
	Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
	pos = nx.spring_layout(Gcc)
	plt.axis('off')
	nx.draw_networkx_nodes(Gcc, pos, node_size=20)
	nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
	plt.savefig(base + 'degree_rank_connected.pdf')
	plt.close()

def random_walk(G, walkLength):
	pos=nx.spring_layout(G)
	A = nx.adj_matrix(G)
	A = A.todense()
	A = numpy.array(A, dtype = numpy.float64)
	# let's evaluate the degree matrix D
	D = numpy.diag(numpy.sum(A, axis=0))
	# ...and the transition matrix T
	T = numpy.dot(numpy.linalg.inv(D),A)
	# define the starting node, say the 0-th
	p = numpy.array([1]+ [0] * (len(G)-1)).reshape(-1,1)
	visited = list()
	for k in range(walkLength):
	    # evaluate the next state vector
	    p = numpy.dot(T,p)
	    # choose the node with higher probability as the visited node
	    visited.append(numpy.argmax(p))
	#nx.draw(G)  # networkx draw()
	#print(visited)
	#nx.draw_networkx_nodes(G,pos,nodelist=visited,node_color='b',node_size=500,alpha=0.8)
	#plt.draw()
	#plt.savefig('network.png') 
	#print(len(G), ' vs ', len(visited))


def get_top_nodes(G, percent, criteria):
	#print('Removing nodes (CRITERIA: %s) ....'%criteria)
	if criteria == 'betweenness':
		top_nodes = get_top_betweenness(G,percent)
	if criteria == 'rank':
		top_nodes = get_top_voterank(G, percent)
	if criteria =='katz':
		top_nodes = get_top_katzcentrality(G, percent)
	if criteria =='closeness':
		top_nodes = get_top_closeness(G, percent)
	if criteria =='random':
		top_nodes = get_top_random(G, percent)
	return top_nodes
	

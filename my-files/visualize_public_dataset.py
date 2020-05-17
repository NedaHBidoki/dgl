import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import sys
import dgl
from dgl.data import citation_graph as citegrh

#data = citegrh.load_cora()
data = citegrh.load_citeseer()
G = dgl.DGLGraph(data.graph)
labels = th.tensor(data.labels)

def visualize(labels, g):
    pos = nx.random_layout(g)#, seed=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    plt.savefig('citeseer4.png')
visualize(labels, G.to_networkx())

sys.exit()

# find all the nodes labeled with class 0
label0_nodes = th.nonzero(labels == 0).squeeze()
# find all the edges pointing to class 0 nodes
src, _ = G.in_edges(label0_nodes)
src_labels = labels[src]
# find all the edges whose both endpoints are in class 0
intra_src = th.nonzero(src_labels == 0)
print('Intra-class edges percent: %.4f' % (len(intra_src) / len(src_labels)))
train_set = dgl.data.CoraBinary()
G1, pmpd1, label1 = train_set[1]
nx_G1 = G1.to_networkx()

def visualize(labels, g):
    pos = nx.spring_layout(g, seed=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    plt.savefig('test.png')
visualize(label1, nx_G1)

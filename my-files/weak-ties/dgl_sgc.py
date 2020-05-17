import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx

import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import pandas as pd

import network_analytics as na
import metrics as mt
import time
from numpy.linalg import matrix_power
class SGCLayer(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 out_feats,
                 K=2):
        super(SGCLayer, self).__init__()
        self.g = g
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.K = K
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h):
        h = torch.mm(h, self.weight)
        for _ in range(self.K):
            # normalization by square root of src degree
            #print("np.shape(self.g.ndata['norm']):",np.shape(self.g.ndata['norm']))
            #print("np.shape(h):",np.shape(h))
            #print("type(h):",type(h))
            h = h * self.g.ndata['norm']
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'))
            h = self.g.ndata.pop('h')
            # normalization by square root of dst degree
            h = h * self.g.ndata['norm']
        return h

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return (correct.item() * 1.0 / len(labels)),indices, labels

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_weak_ties_network(dgl_g,node_list):
	G = dgl_g.to_networkx()
	A = nx.to_numpy_matrix(G, nodelist = node_list)
	#print('sum(np.diagonal):',sum(np.diagonal(A)))
	#print('sum(np.diagonal):',(np.sum(A)))
	print('Is symmetric?:', check_symmetric(A))
	np.fill_diagonal(A, 0)
	A2 = matrix_power(A,2)
	np.fill_diagonal(A2, 0)
	A2 = A2>0
	A2 = A2 + A
	A2 = A2>0
	#print('sum(np.diagonal):',np.sum(A2))
	G = nx.from_numpy_matrix(A2)
		
	temp_G = G.to_undirected()
	'''
	if (dgl_g.number_of_nodes() != G.number_of_nodes()):
		print('A shape:',np.shape(A))
		print('A2 shape:',np.shape(A2))
		print('number of nodes in weak tie network:',G.number_of_nodes())
		print('number of nodes in strong tie network:',dgl_g.number_of_nodes())
		print('number of nodes in undirected weak tie network:',temp_G.number_of_nodes())
		print('number of edges in weak tie network:',(G.number_of_edges()))
		print('number of edges in strong tie network:',(dgl_g.number_of_edges()))
		print('number of edges in undirected weak tie network:',(temp_G.number_of_edges()))
		print('#######################')
	'''
	return G, temp_G

def main(args):
    #print('in main...')
    percents = [10,20,30,40]#,5, 10,15,20,25,30,35,40,45,50,55,60,65,70]##[1,2,3,4,5,6,7,8]
    criteria = ['random']#'closeness','rank','betweenness']#'random']#]#, 'katz' ,
    degrees =[0,1,2,3,4,5,6,7]#]
    folds =range(160,162)
    epocs = [600]#,200]

    # load and preprocess dataset
    data = load_data(args)
    dgl_g = DGLGraph(data.graph)
    features, labels = data.features, data.labels

    #####################  MY WORK ################
    #g = DGLGraph(data.graph)
    #print(g.ndata['norm'] )
    #sys.exit()
    print(np.shape(data.features))
    print(np.shape(data.labels))
    #print(type(data.graph))
    
    node_list = dgl_g.nodes().tolist()
    G, temp_G = get_weak_ties_network(dgl_g,node_list)
    

    for p in percents:
        for c in criteria:
            G_ = G.copy()
            g, remaining_nodes= na.network_preprocess(G_, temp_G, node_list, args.dir_, args.dataset, p, c)
            for f in folds:
                features2,labels2, train_mask,val_mask,test_mask = na.get_model_parameters(remaining_nodes, features, labels)
                #print('number of nodes in strong tie network:',dgl_g.number_of_nodes())
                #print('number of nodes in weak tie network:',G.number_of_nodes())
                #print('number of nodes in undirected weak tie network:',temp_G.number_of_nodes())
    
                #print('number of nodes in weak tie network after %s removal:'%p,len(remaining_nodes))
                #print('number of nodes in weak tie network after %s removal:'%p,g.number_of_nodes())
                #print('number of features in weak tie network after %s removal:'%p,len(features2))

                for d in degrees:
                    print('******************* degree %s******************'%d)
                    for e in epocs:
                        print(args.dataset,'criteria:',c,'percent:',p, 'degree:',d,'fold:',f,'epoch:',e)
                        run_(data, args, g, features2, labels2, train_mask,val_mask,test_mask, p, c, f, d, e)


def run_(data, args, g, features, labels, train_mask,val_mask,test_mask, percent, criteria, fold, degree,n_epochs):
    
    #dir_, dataset, percent, criteria = 'test_plots/', 'test', '1','betweenness'
    #g,features,labels,train_mask,val_mask,test_mask = na.network_preprocess(G, temp_G, node_list, features, labels, args.dir_, args.dataset, percent, criteria)
    #print('train_mask:',sum(train_mask))
    #print('val_mask:',sum(val_mask))
    #print('test_mask:',sum(test_mask))
    features = torch.FloatTensor(features) 
    labels = torch.LongTensor(labels)  
    train_mask = torch.ByteTensor(train_mask)
    val_mask = torch.ByteTensor(val_mask)  
    test_mask = torch.ByteTensor(test_mask)  
    in_feats = features.shape[1] 
    n_classes = data.num_labels  
    n_edges = g.number_of_edges()  
    n_nodes = g.number_of_nodes() 


    ##############################################
    '''
    features = torch.FloatTensor(data.features) #***************** commented ******************
    labels = torch.LongTensor(data.labels)  #***************** commented ******************
    train_mask = torch.ByteTensor(data.train_mask)  #***************** commented ******************
    val_mask = torch.ByteTensor(data.val_mask)  #***************** commented ******************
    test_mask = torch.ByteTensor(data.test_mask)  #***************** commented ******************
    in_feats = features.shape[1]  #***************** commented ******************
    n_classes = data.num_labels  #***************** commented ******************
    n_edges = data.graph.number_of_edges()  #***************** commented ******************
    n_nodes = data.graph.number_of_nodes()  #***************** commented ******************
    '''
    print("""----Data statistics------'
      #Nodes %d
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_nodes,n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # graph preprocess and calculate normalization factor
    '''
    g = DGLGraph(data.graph) #***************** commented ******************
    '''
    n_edges = g.number_of_edges()
    # add self loop
    g.add_edges(g.nodes(), g.nodes())  ##@@
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create SGC model
    model = SGCLayer(g,
                in_feats,
                n_classes,
                K=degree)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc , output , preds = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    acc , output , preds = evaluate(model, features, labels, test_mask)

    print("Test Accuracy {:.4f}".format(acc))
    micro, macro, fpr, tpr, threshold,roc_auc, acc_, prc_mac, prc_mic, prc_wei, tn, fp, fn, tp = mt.f1(output.tolist(), preds.tolist())
    with open("%s/%s_results_deg2.csv"%(args.dir_, criteria), "a") as myfile:
        myfile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(args.dataset, micro, macro, fpr, tpr, threshold,roc_auc, acc_, prc_mac, prc_mic, prc_wei, tn, fp, fn, tp,criteria,percent, fold, degree))
    print(args.dataset, micro, macro, fpr, tpr, threshold,roc_auc, acc_, prc_mac, prc_mic, prc_wei, tn, fp, fn, tp,criteria,percent, fold,degree)
if __name__ == '__main__':
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

    main(args)

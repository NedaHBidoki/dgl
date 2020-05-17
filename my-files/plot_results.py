import pandas as pd
import sys
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os
from collections import Counter
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import csv
from ast import literal_eval

plt.rc('font', family='serif')


def plot_acc(df,save_dir,file_):
	fig, ax = plt.subplots()
	df = df.groupby(['dataset','criteria', 'percent'],as_index=False).mean()
	gr = df.groupby(['dataset','criteria']).plot(x='percent',y='acc_', ax=ax, legend=False, color='black', linestyle='--')#,style=['+-','o-','.--'], , markevery=100, marker='o', markerfacecolor='black')#.unstack(level=-1)['acc_'].plot()
	ax.set_title("Accuracy vs Percentage of Removed Nodes", fontsize=12);
	ax.set_xlabel("% of nodes", fontsize=12)
	ax.set_ylabel("Accuracy", fontsize=12)
	ax.grid()
	plt.savefig('%s.pdf'%file_.split('.')[0])
	plt.show()

def plot_acc_merged(df,save_dir, dbs, criteria):
	symbols = {'betweenness':'*','rank':'o','closeness':'<', 'random':'4'}
	#colors = {'citeseer':'r','syn':'b','cora':'y','pubmed':'g'}
	colors = {'betweenness':'r','rank':'b','closeness':'y','random':'g'}
	metrics =  ['macro' ,'micro' ,'acc_' ,'prc_mic' , 'prc_mac'] # , 
	met_dic = {'prc_mac':'Precision Macro', 'acc_':'Accuracy', 'prc_mic':'Precision Micro', 'micro':'F1 micro', 'macro':'F1 macro'}
	for metric in metrics:
		for d in dbs:
			for c in criteria:
				df1 = df[(df['dataset']==d) & (df['criteria']==c) & (df['percent'].isin([0,10,20,30,40]))]
				df1.dropna()
				df_mean = df1.groupby(['dataset','criteria', 'percent'],as_index=False)[metric].mean()
				#sys.exit()
				df_std= df1.groupby(['dataset','criteria', 'percent'],as_index=True).std()
				print(type(df_std))
				plt.grid()
				x = df_mean['percent'].values
				mea = df_mean[metric].values
				std = df_std[metric].values
				#print('std:',std)
				plt.fill_between(x, mea - std,mea + std, alpha=0.3,   color=colors[c])
				plt.plot(x, mea, marker=symbols[c], color="black", label=c.capitalize() )
				plt.title("Comparison of Removal Metrics in '%s' Dataset"%d.capitalize(), fontsize=11);
				plt.xlabel("% of Network Nodes", fontsize=12)
				plt.ylabel(met_dic[metric], fontsize=12)
				plt.legend()
				plt.grid(True)
			plt.savefig('final_plots/%s_%s.pdf'%(d,metric))
			#plt.show()
			plt.close()

def merge_and_plot(save_dir,dbs, criteria):
	columns = ['dataset', 'micro', 'macro', 'fpr', 'tpr', 'threshold', 'roc_auc', 'acc_', 'prc_mac', 'prc_mic', 'prc_wei', 'TN', 'FP', 'FN', 'TP', 'criteria', 'percent', 'fold']
	df = pd.DataFrame()	
	for c in criteria:
		file_ = '%s_results.csv'%c
		df1 = pd.read_csv(save_dir + file_, names = columns, sep='\t')
		df = df.append(df1, ignore_index = True)
	plot_acc_merged(df,save_dir, dbs,criteria )
	return df

def get_deg_comp_comm_data(save_dir,dbs,criteria,percents,degrees):
	fls = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
	df_degrees = pd.DataFrame()
	df_component = pd.DataFrame()
	df_communities = pd.DataFrame()
	for f in fls:
		with open(os.path.join(save_dir, f), 'rb') as file_:
			reader = csv.reader(file_)
			degrees = list(reader)
			flat_list = [item for sublist in degrees for item in sublist]
			#print('degrees:',degrees)
			d = f.split('_')[0]
			p = f.split('_')[1]
			c = f.split('_')[2]
			fo = f.split('_')[3]
			dic_ = {'dataset':d,'percent':p,'criteria':c,'fold':fo,'list':flat_list}
		if 'nodes_degree' in f:
			df_degrees = df_degrees.append(dic_, ignore_index=True)
		elif 'component' in f:
			df_component = df_component.append(dic_, ignore_index=True)
		elif 'communities' in f:
			df_communities = df_communities.append(dic_, ignore_index=True)
	df_degrees.to_csv(save_dir+'/aggregate_data/df_degrees.csv')
	df_component.to_csv(save_dir+'/aggregate_data/df_component.csv')
	df_communities.to_csv(save_dir+'/aggregate_data/df_communities.csv')
	return df_degrees, df_component, df_communities

def plot_degrees(file_, dbs, criteria, percents):
	percents = [5,25,50,65]
	markers = {'5':'*','25':'o','65':'^','50':'<'}
	colors = {'5':'r','25':'b','65':'y','50':'g'}
	df = pd.read_csv(file_)
	for d in dbs:
		for c in criteria:
			df1 = df[(df['dataset']==d) & (df['criteria']==c)]
			print('len df1',len(df1))
			y = df1['percent'].unique().tolist()
			print('y:', y)
			for i,p in enumerate(percents):
				#degree_sequence = sorted(df[(df['dataset']==d) & (df['criteria']==c) & (df['percent']==str(p)+'p')]['list'].iloc[0:].values,reverse=True)
				degree_sequence = df[(df['dataset']==d) & (df['criteria']==c) & (df['percent']==str(p)+'p')]['list'].iloc[0]
				degree_sequence = sorted(list(map(int, literal_eval(degree_sequence)) ),reverse=True)

				print('degrees',degree_sequence)
				print('p:',p)
				plt.loglog(degree_sequence, color=colors[str(p)], marker=markers[str(p)],label=p, alpha=.3)# bins='auto',color='gray')
				#plt.show() str(0.6*i)
			plt.title(" '%s' degree distributions from '%s' rank node removal"%(d.capitalize(), c),fontsize=12)


			plt.ylabel("Degree",fontsize=12)
			plt.xlabel("Rank",fontsize=12)
			plt.legend()
			plt.grid(True)
			plt.savefig('final_plots/%s_%s_node_degrees.pdf'%(d,c))
			#plt.show()
			plt.close()
def get_net_info_data(save_dir,dbs,criteria,percents,degrees):
	fls = [f for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
	df = pd.DataFrame()
	for f in fls:
		with open(os.path.join(save_dir, f), 'rb') as file_:
			reader = csv.reader(file_)
			values = list(reader)
		#print('values',values)
		density = float(values[0][1])
		lc_diameter = float(values[1][1])
		triadic_closure = float(values[2][1])
		#print('density',density)
		#print('lc_diameter',lc_diameter)
		#print('triadic_closure',triadic_closure)
		d = f.split('_')[0]
		p = f.split('_')[1][:-1]
		c = f.split('_')[2]
		fo = f.split('_')[3]
		dic_ = {'dataset':d,'percent':p,'criteria':c,'fold':fo,'density':density,'lc diameter':lc_diameter, 'triadic closure':triadic_closure }
		df = df.append(dic_, ignore_index=True)
	df.to_csv(save_dir+'/../aggregate_data/df_net_info.csv')
	return df

def plot_net_info(file_, dbs, criteria, percents):
	df = pd.read_csv(file_)
	symbols = {'betweenness':'*','rank':'o','closeness':'<','random':'4'}
	linestyles = {'betweenness':'dashed','rank':'dashdot','closeness':'solid'}
	colors = {'betweenness':'r','rank':'b','closeness':'y'}
	features = ['lc diameter','density','triadic closure']#
	for f in features:
		for d in dbs:
			for c in criteria:
				df1 = df[(df['dataset']==d) & (df['criteria']==c)]
				df1.dropna()
				df_mean = df1.groupby(['dataset','criteria', 'percent'],as_index=False)[f].mean()
				#sys.exit()
				df_std= df1.groupby(['dataset','criteria', 'percent'],as_index=True).std()
				print((df_std))
				plt.grid()
				x = df_mean['percent'].values
				mea = df_mean[f].values
				std = df_std[f].values
				#print('std:',std)
				plt.fill_between(x, mea - std,mea + std, alpha=0.3,   color=colors[c])
				plt.plot(x, mea, marker=symbols[c], color=colors[c], label=c.capitalize(), linestyle= linestyles[c])
				plt.title("'%s' %s using node '%s' removal"%(d.capitalize(),f,c), fontsize=12);

				plt.xlabel("% of Network Nodes", fontsize=12)
				plt.ylabel(f.capitalize(), fontsize=12)
				plt.legend()
				plt.grid(True)
			plt.savefig('final_plots/%s_%s.pdf'%(d,f))
			#plt.show()
			plt.close()

#### plotting F1 vs epoch  #####
def plot_f1_k(dbs, criteria,plots_dir, aws_result_dir, careAI_result_dir, newton_result_dir):
	from matplotlib.pyplot import cm
	metrics =  ['acc_' ,'macro' ,'micro' ,'prc_mic' , 'prc_mac'] # , 
	met_dic = {'prc_mac':'Precision Macro', 'acc_':'Accuracy', 'prc_mic':'Precision Micro', 'micro':'F1 micro', 'macro':'F1 macro'}
	columns = ['dataset', 'micro', 'macro', 'fpr', 'tpr', 'threshold', 'roc_auc', 'acc_', 'prc_mac', 'prc_mic', 'prc_wei', 'TN', 'FP', 'FN', 'TP', 'criteria', 'percent','fold','k']
	colors ={10:'r',15:'b',20:'y',25:'y',30:'darkblue',35:'black',40:'b',50:'olive'}
	markers ={10:'*',15:'^',20:'o',25:'p',30:'>',35:'8',40:'s',50:'4'}
	styles ={10:'dashed',15:'dashdot',20:'solid',25:'dotted',30:'dashed',35:'dotted',40:'solid',50:'solid'}
	#print(df['k'])
	color=iter(cm.rainbow(np.linspace(0,1,20)))
	for c in criteria:
		f = aws_result_dir + '%s_results_deg2.csv'%c
		df = pd.read_csv(f,names=columns, sep='\t')
		df = df[(df['k'].notnull()) & (df['k']<8)]
		print(f,len(df))
		f1 = careAI_result_dir + '%s_results_deg2.csv'%c
		df1 = pd.read_csv(f1,names=columns, sep='\t')
		df1 = df1[(df1['k'].notnull()) & (df1['k']<8)]
		df = df.append(df1, ignore_index = True) 
		print(f1,len(df1))
		print(len(df))
		f2 = newton_result_dir + '%s_results_deg2.csv'%c
		df2 = pd.read_csv(f2,names=columns, sep='\t')
		df2 = df2[(df2['k'].notnull()) & (df2['k']<8)]
		df= df.append(df2, ignore_index = True) 
		print(f2,len(df2))
		print(len(df))
		#print(df1.head())
		for d in dbs:
			for m in metrics:
				for p in [10,20,40]:#set(df['percent'].values.tolist()):#[10,25,40]:#
					print('db',d,'p',p,'c',c,'m',m)
					df1 = df[(df['dataset']==d) & (df['criteria']==c) & (df['percent']==p)]
					print(len(df1))
					if len(df1)==0:
						continue
					#df1.groupby(['k']).mean()[m].plot.bar(yerr=df1.groupby(['k']).mean(),color=next(color), grid=True, label=p)
					#df1.plot(edgecolor='b', color='y', grid=True, hatch="*",width=0.3, linewidth=3)
					df1 = df1.groupby(['k'])[m].mean().plot(label=p,color=colors[p],marker=markers[p],linestyle=styles[p])
					#print(df1.head())

					'''
					#### plotting 3D
					threedee = plt.figure().gca(projection='3d')
					threedee.scatter(df1['F1'], df1['bin'], df1['epoch'])
					threedee.set_xlabel('F1')
					threedee.set_ylabel('bin')
					threedee.set_zlabel('epoch')
					'''
					plt.grid(True)
					plt.xticks(rotation='vertical')
					plt.xlabel('K',fontsize=11)
					plt.ylabel('%s'%(met_dic[m]),fontsize=11)
					plt.tight_layout(True)
					plt.title('%s %s '%(d,c))
					#plt.legend()
					#plt.tight_layout()
				#plt.ylim(top=0.9)  # adjust the top leaving bottom unchanged
				#plt.ylim(bottom=0.7)  # adjust the bottom leaving top unchanged

				plt.legend()
				plt.savefig('%s/%s_%s_%s.pdf'%(plots_dir,c,d,m))
				#plt.show()
				plt.close()
				#sys.exit()
			#plt.close()
def plot_weak_ties_f1_k(dbs, criteria,plots_dir, dir_,method):
	metrics =  ['acc_' ,'macro' ,'micro' ,'prc_mic' , 'prc_mac'] # , 
	met_dic = {'prc_mac':'Precision Macro', 'acc_':'Accuracy', 'prc_mic':'Precision Micro', 'micro':'F1 micro', 'macro':'F1 macro'}
	columns = ['dataset', 'micro', 'macro', 'fpr', 'tpr', 'threshold', 'roc_auc', 'acc_', 'prc_mac', 'prc_mic', 'prc_wei', 'TN', 'FP', 'FN', 'TP', 'criteria', 'percent','fold','k']
	colors ={10:'r',15:'b',20:'y',25:'y',30:'darkblue',35:'black',40:'gold',50:'olive'}
	markers ={10:'*',15:'^',20:'o',25:'p',30:'>',35:'8',40:'s',50:'4'}
	styles ={10:'dashed',15:'dashdot',20:'solid',25:'dotted',30:'dashed',35:'dotted',40:'solid',50:'solid'}
	#print(df['k'])
	for c in criteria:
		f = dir_ + '%s_results_deg2.csv'%c
		df = pd.read_csv(f,names=columns, sep='\t')
		df = df[(df['k'].notnull()) & (df['k']<8)]
		for d in dbs:
			for m in metrics:
				for p in [10,20,40]:#set(df['percent'].values.tolist()):#[10,25,40]:#
					print('db',d,'p',p,'c',c,'m',m)
					df1 = df[(df['dataset']==d) & (df['criteria']==c) & (df['percent']==p)]
					print(len(df1))
					if len(df1)==0:
						continue
					#df1.groupby(['k']).mean()[m].plot.bar(yerr=df1.groupby(['k']).mean(),color=next(color), grid=True, label=p)
					#df1.plot(edgecolor='b', color='y', grid=True, hatch="*",width=0.3, linewidth=3)
					df1 = df1.groupby(['k'])[m].mean().plot(label=p,color=colors[p],marker=markers[p],linestyle=styles[p])
					#print(df1.head())

					'''
					#### plotting 3D
					threedee = plt.figure().gca(projection='3d')
					threedee.scatter(df1['F1'], df1['bin'], df1['epoch'])
					threedee.set_xlabel('F1')
					threedee.set_ylabel('bin')
					threedee.set_zlabel('epoch')
					'''
					plt.grid(True)
					plt.xticks(rotation='vertical')
					plt.xlabel('K',fontsize=11)
					plt.ylabel('%s'%(met_dic[m]),fontsize=11)
					plt.tight_layout(True)
					plt.title('%s %s method: %s'%(d,c,method))
					#plt.legend()
					#plt.tight_layout()
				#plt.ylim(top=0.9)  # adjust the top leaving bottom unchanged
				#plt.ylim(bottom=0.7)  # adjust the bottom leaving top unchanged

				plt.legend()
				plt.savefig('%s/%s_%s_%s.pdf'%(plots_dir,c,d,m))
				#plt.show()
				plt.close()
				#sys.exit()
			#plt.close()

def main():

	dbs = ['cora','citeseer']#,'syn']#,], ,'pubmed' ,'syn',
	dir_ = 'results/'
	percents = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]##[1,2,3,4,5,6,7,8]
	criteria = ['rank','betweenness','closeness']#'rank']#, #, 'katz'
	degrees =[2]
	folds =range(5)
	epocs = [200]
	base ='/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_public_data/'
	save_dir = base +'results/'
	merge_and_plot(save_dir,dbs, criteria)
	sys.exit()
	#df_degrees, df_component, df_communities = get_deg_comp_comm_data(save_dir+'/files/',dbs,criteria,percents,degrees)
	
	#plot_degrees(save_dir+'/files/aggregate_data/df_degrees.csv', dbs, criteria, percents)
	#get_net_info_data(save_dir+'/files/net_info/',dbs,criteria,percents,degrees)
	#plot_net_info(save_dir+'/files/aggregate_data/df_net_info.csv', dbs, criteria, percents)
	
	
	### plot k diagrams....
	plots_dir = base + 'final_plots/A/k'
	aws_result_dir = base +'results/'
	careAI_result_dir = base + 'careAI_results/'
	newton_result_dir = base + 'Newton_results/'
	plot_f1_k(dbs, criteria,plots_dir, aws_result_dir, careAI_result_dir, newton_result_dir )
	####
	
	sys.exit()


	#### plot weak ties diagrams
	method = 'A^2-I'
	a = 'strong and weak ties with betweenness removal'
	plots_dir = base + 'final_plots/%s/'%method
	if not os.path.exists(plots_dir):
		os.makedirs(plots_dir)
	dir_ = base +'results_weak_ties/%s/'%method
	plot_weak_ties_f1_k(dbs, criteria,plots_dir, dir_,method)

main()

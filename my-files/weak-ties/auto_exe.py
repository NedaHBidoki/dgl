import time
import os

def main():
	dbs = ['citeseer','cora']#,]#,]]#, ,'pubmed' 
	dir_ = 'results/'
	start_time1 = time.time()
	method = 'A_test'
	for db in dbs:	
		#print('INFO: --dataset %s --criteria %s --degree %s --fold %s --n-epochs %s --dir_ %s' %(db, c, d, f, e, dir_))					
		start_time2 = time.time()
		os.system('python3 -W ignore dgl_sgc.py --dataset %s --method %s --dir_ %s' %(db, method, dir_))
		print("--- this iteration %s seconds ---" % (time.time() - start_time2))
	print("--- %s seconds ---" % (time.time() - start_time1))

main()

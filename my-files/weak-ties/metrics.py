from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import time
from sklearn.metrics import confusion_matrix
import os 
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(preds, labels):
    #preds = output.max(1)[1]
    #preds = preds.cpu().detach().numpy()
    #labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    try:
        fpr, tpr, threshold = roc(labels, preds)
        roc_auc = metrics.auc(fpr, tpr)
    except:
        fpr, tpr, threshold,roc_auc= 'nan','nan','nan','nan'
    acc_ = acc(labels, preds)
    prc_mac, prc_mic, prc_wei = precision(labels, preds)
    tn, fp, fn, tp = conf_matrix(labels, preds)
    return micro, macro, fpr, tpr, threshold,roc_auc, acc_, prc_mac, prc_mic, prc_wei, tn, fp, fn, tp

def acc(labels, preds):
	return metrics.accuracy_score(labels, preds)

def precision(labels, preds):
	prc_mac = metrics.precision_score(labels, preds, average='macro', zero_division=1)
	prc_mic = metrics.precision_score(labels, preds, average='micro', zero_division=1)
	prc_wei = metrics.precision_score(labels, preds, average='weighted', zero_division=1)
	return prc_mac, prc_mic, prc_wei

def conf_matrix(labels, preds):
	tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
	return tn, fp, fn, tp

def roc(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot(fpr, tpr, 'b')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


    #fpr, tpr, _ = metrics.roc_curve(y_test,  preds)
    #auc = metrics.roc_auc_score(y_test, preds)
    #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    #plt.legend(loc=4)
    save_dir = 'plots/test/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('%s/roc_%s_%s.pdf'%(save_dir,time.time()))
    return fpr, tpr, threshold

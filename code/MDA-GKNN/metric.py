from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from collections import Counter
from copy import deepcopy

def calc_f1(y_true, y_pred, is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_metrics(y_true, y_pred, is_sigmoid):
    y_pred_prob = y_pred.reshape(-1)
    
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    prec_reca_f1_supp_report = classification_report(y_true, y_pred, target_names = ['label 0', 'label 1'])
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    pos_acc = tp/sum(y_true).item()
    neg_acc = tn/(len(y_pred)-sum(y_pred).item()) # [y_true=0 & y_pred=0] / y_pred=0

    roc_auc = roc_auc_score(y_true, y_pred_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(reca, prec)

    return prec_reca_f1_supp_report, pos_acc, neg_acc, roc_auc, aupr

def metrics(y_true, y_prob, is_sigmoid, isprint = True):

    y_pred = deepcopy(y_prob)
    
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    
    #print(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    pos_acc = tp / sum(y_true)
    neg_acc = tn / (len(y_pred) - sum(y_pred)) # [y_true=0 & y_pred=0] / y_pred=0
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    
    recall = tp / (tp+fn)
    precision = tp / (tp+fp)
    f1 = 2*precision*recall / (precision+recall)
    
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    
    if isprint:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))

    return (y_true, y_pred, y_prob), (accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc)
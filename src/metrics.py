import json
import sys
import numpy as np
import json
import re
import math
import nltk
import argparse
import eval

from scipy.stats import kendalltau 
from tqdm import tqdm


def kendall_tau(order, ground_truth):
    """
    Computes the kendall's tau metric 
    between the predicted sentence order and true order
    Input: 
            order: list of ints denoting the predicted output order
            ground_truth: list of ints denoting the true sentence order
    
    Returns:
            kendall's tau - float
    """
    
    if len(ground_truth) == 1:
        if ground_truth[0] == order[0]:
            return 1.0
        
    reorder_dict = {}
        
    for i in range(len(ground_truth)):
        reorder_dict[ground_truth[i]] = i
        
    new_order = [0] * len(order)
    for i in range(len(new_order)):
        if order[i] in reorder_dict.keys():
            new_order[i] = reorder_dict[order[i]]
    
    corr, _ = kendalltau(new_order, list(range(len(order))))
    return corr

def lcs(X , Y): 
    """
    Computes the longest common subsequence between two sequences
    Input:
            X: list of ints
            Y: list of ints
    
    Returns:
            LCS: int
    """
    m = len(X) 
    n = len(Y) 

    L = [[None]*(n+1) for i in range(m+1)] 

    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 

    return L[m][n] 


def skip_bigrams(arr):
    """
    Utility function for Rouge-S metric
    """
    bigrams = set()
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            bigrams.add((arr[i], arr[j]))
    return bigrams

def rouge_s(gold, pred):
    """
    Rouge-S metric between two sequence
    Input:
            gold: list of ints
            pred: list of ints
    
    Returns:
            Rouge-S score
    """

    if len(gold) == 1 or len(pred) == 1:
        return int(gold[0] == pred[0])
    
    gold_bigrams = skip_bigrams(gold)
    pred_bigrams = skip_bigrams(pred)
    
    total = len(gold_bigrams)
    same = len(gold_bigrams.intersection(pred_bigrams))
    return (same / total)


def clean_output(gold, predictions):
    """
    Utility function to clean generated output from BART
    """

    label = gold.replace("<eos>", "").strip()
    labels = [int(id_[2:-1]) for id_ in label.split()]
    
    # handle cases when output is empty
    if len(predictions) == 0:
        return labels, []
    
    preds = []
    for p in predictions[0].split():
        pos = re.findall('\\d+', p)
        if len(pos) == 1:
            preds.append(int(pos[0]))
    return labels, preds


def acc_compare(gold, predict):
    return np.sum((np.array(gold) == np.array(predict)))


def base_compare(gold, predict, outline):
    ipt = ["#".join(g) for g in outline]
    truth = gold
    pred = predict

    kw2id = []
    for i1, t1 in zip(ipt, truth):
        kw_list = i1.strip().split("#")
        pos = [t1.strip().find(kw.strip()) for kw in kw_list]

        idlist = list(range(len(pos)))
        orderlist = sorted(idlist, key=lambda x: pos[x])
        kw2id.append({})
        for idl, ord in zip(idlist, orderlist):
            kw2id[-1][kw_list[ord]] = idl


    eval_data = [{"reference": eval.proline(g), "candidate": eval.proline(p)} for g, p in zip(gold, predict)]
    res = eval.bleu(eval_data)
    res.update(eval.repetition_distinct(eval_data))
    res.update(eval.rouge(ipt=ipt, cand=pred))
    res.update(eval.order(ipt=ipt, cand=pred, kw2id=kw2id))
    return res


def overall_compare(res):
    small = [26.58, 16.04, 17.90, 31.38, 83.64, 63.15]
    true = [100, 100, 23.47, 42.17, 100, 100]
    predict = [res["bleu-1"], res["bleu-2"], res["distinct-3"], res["distinct-4"], res["coverage"], res["order"]]
    sum = 0
    wi = []
    for a, b in zip(true, small):
        wi.append(a / b)
        sum += a / b
    overall = 0
    for a, w in zip(predict, wi):
        overall += a * (w / sum)
    return overall

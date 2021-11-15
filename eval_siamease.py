from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import sys
import numpy as np 
import random
import math

def computeROC_AUC(OGs_HDs_sim,HDs_keys,OGs_keys,h_positives):
    OGs_HDs_sim_t = OGs_HDs_sim
    ranks = np.argsort(OGs_HDs_sim_t, axis=1)
    TPR = [0]
    FPR = [0]
    prev= [0,0]
    P=0
    N=0
    positive_genes = set([])
    includedDiseases =  np.zeros(len(HDs_keys))
    print(OGs_HDs_sim_t.shape,len(HDs_keys),len(OGs_keys))

    positives_matrix = np.zeros([len(HDs_keys),len(OGs_keys)])
    for og in range(0,len(OGs_keys)):
        for hd in range(0,len(HDs_keys)):
            if(HDs_keys[hd] in  h_positives):
                if(OGs_keys[og] in h_positives[HDs_keys[hd]]):
                    positives_matrix[hd][og] = 1
                    includedDiseases[hd]=1
                    positive_genes.add(OGs_keys[og])
    ranking_dic_disease={}
    ranking_dic_gene={}
    positives_ranks = {}

    for hd in range(0,len(HDs_keys)):
        if(includedDiseases[hd]==1):
            p = np.sum(positives_matrix[hd])
            P+= p
            N+= len(OGs_keys)-p
    print("p",P)    
    print("included_Disease", np.sum(includedDiseases))
    for r in range(0,len(OGs_keys)):
        TP = prev[0]
        FP = prev[1]
        for hd in range(0,len(HDs_keys)):
            if(includedDiseases[hd]==1):
                g=len(OGs_keys)-r-1
                if (positives_matrix[hd,ranks[hd][g]] == 1):
                    TP+=1
                    if(HDs_keys[hd] not in ranking_dic_disease):
                        ranking_dic_disease[HDs_keys[hd]] = r+1
                        if (OGs_keys[g] not in ranking_dic_gene):
                            ranking_dic_gene[OGs_keys[g]]=[]
                        ranking_dic_gene[OGs_keys[g]].append(r+1)
                else:
                    FP+=1
        prev = [TP,FP]
        TPR.append(TP/P)
        FPR.append(FP/N)
    return TPR,FPR,np.trapz(TPR,FPR),P,N

diseases_keys = set([])
genese_keys = set([])

scores = {}

for fold in range(10):
    print("starting fold",fold)
    with open("out"+str(fold)+".txt") as f:
        content = f.readlines()
    for line in content:
        if '|' not in line:
            ls = line.split()
            if(len(ls)>2):
#                 print(ls)
                omim = ls[0]
                mgi = ls[1]
                score = float(ls[2])
                scores[(omim,mgi)] = score
                diseases_keys.add(omim)
                genese_keys.add(mgi)

ls_diseases = list(diseases_keys)
ls_genes = list(genese_keys)

print("calculating the scores_matrex")
scores_matrex = np.empty((len(ls_diseases),len(ls_genes)))
for i in range(len(ls_diseases)):
    for j in range(len(ls_genes)):
        scores_matrex[i][j] = scores[ls_diseases[i],ls_genes[j]]


positives = {}
with open('m_positives_m.json','r') as f:
    positives = json.load(f)


print("Evaluating")
TPR,FPR,ROC,P,N = computeROC_AUC(scores_matrex,ls_diseases,ls_genes,positives)



#-----------------------------------------------------------
def precisionAt(TPR,P,FPR,N,at):
    return (TPR[at]*P)/((TPR[at]*P)+(FPR[at]*N))

def recallAt(TPR,P,FPR,N,at):
    return TPR[at]

def hitAt(TPR,P,FPR,N,at):
    return TPR[at]*P
#---------------------------------------------------------


print("ROC_AUC = ",ROC)
print("CI = ",2*(math. sqrt(ROC*(1-ROC)/(min([P,N])))))
print("Precision@1 @10 @50 @100 = ",precisionAt(TPR,P,FPR,N,1),precisionAt(TPR,P,FPR,N,10),precisionAt(TPR,P,FPR,N,50),precisionAt(TPR,P,FPR,N,100))
print("Recall@1 @10 @50 @100 = ",recallAt(TPR,P,FPR,N,1),recallAt(TPR,P,FPR,N,10),recallAt(TPR,P,FPR,N,50),recallAt(TPR,P,FPR,N,100))
print("Hits@1 @10 @50 @100 = ",hitAt(TPR,P,FPR,N,1),hitAt(TPR,P,FPR,N,10),hitAt(TPR,P,FPR,N,50),hitAt(TPR,P,FPR,N,100))
print("Total positives = ", P)

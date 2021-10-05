import click as ck
import pandas as pd
from utils import Ontology
import dgl
from dgl import nn as dglnn
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_auc_score, matthews_corrcoef

ORG_ID = '9606'

@ck.command()
@ck.option(
    '--test-inter-file', '-tsif', default=f'data/{ORG_ID}.test_interactions.pkl',
    help='Interactions file (data.py)')
@ck.option(
    '--predictions-file', '-pf', default=f'data/{ORG_ID}.mlp_scores.tsv',
    help='Predictions file with scores')
def main(test_inter_file, predictions_file):
    test_df = pd.read_pickle(test_inter_file)
    labels = test_df['labels'].values
    preds = []
    with open(predictions_file) as f:
        for line in f:
            it = line.strip().split('\t')
            preds.append(float(it[2]))
    preds = np.array(preds)
    roc_auc = roc_auc_score(labels, preds, average='macro')
    print('ROC AUC:', roc_auc)

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


if __name__ == '__main__':
    main()

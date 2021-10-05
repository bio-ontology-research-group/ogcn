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

ORG_ID = '4932'

@ck.command()
@ck.option(
    '--embeddings-file', '-ef', default=f'data/{ORG_ID}.owl2vec.txt',
    help='Predictions file with scores')
@ck.option(
    '--test-inter-file', '-tsif', default=f'data/{ORG_ID}.test_interactions.pkl',
    help='Interactions file (data.py)')
@ck.option(
    '--predictions-file', '-pf', default=f'data/{ORG_ID}.owl2vec_scores.tsv',
    help='Predictions file with scores')
def main(embeddings_file, test_inter_file, predictions_file):
    embeds = {}
    with open(embeddings_file) as f:
        for line in embeds:
            it = line.strip().split()
            eid = it[0]
            emb = list(map(float, it[1:]))
            embeds[eid] = np.array(emb, dtype=np.float32)

    test_df = pd.read_pickle(test_inter_file)
    labels = test_df['labels'].values
    preds = []
    with open(predictions_file, 'w') as f:
        for i, row in enumerate(test_df.itertuples()):
            p1, p2 = row.interactions
            score = cosine_similarity(embeds[p1], embeds[p2])
            f.write(f'{p1}\t{p2}\t{score}\n')
    

def cosine_similarity(a, b):
    nominator = np.dot(a, b)
    a_norm = np.sqrt(np.sum(a**2))
    b_norm = np.sqrt(np.sum(b**2))
    denominator = a_norm * b_norm
    result = nominator / denominator
    return result

if __name__ == '__main__':
    main()

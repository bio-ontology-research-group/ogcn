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
from sklearn.metrics import roc_curve, auc, matthews_corrcoef



@ck.command()
@ck.option(
    '--train-inter-file', '-trif', default='data/4932.train_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--test-inter-file', '-tsif', default='data/4932.test_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--data-file', '-df', default='data/swissprot.pkl',
    help='Data file with protein sequences')
@ck.option(
    '--deepgo-model', '-dm', default='data/deepgoplus.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--model-file', '-mf', default='data/9606.model.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=16,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(train_inter_file, test_inter_file, data_file, deepgo_model, model_file, batch_size, epochs, load):
    device = 'cuda'
    g, annots, prot_idx = load_graph_data(data_file)
    g = g.to(device)
    annots = th.FloatTensor(annots).to(device)
    train_df, test_df = load_ppi_data(train_inter_file, test_inter_file)
    model = PPIModel()
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_labels = th.FloatTensor(train_df['labels'].values).to(device)
    test_labels = th.FloatTensor(test_df['labels'].values).to(device)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        with ck.progressbar(train_df.itertuples(), length=len(train_df)) as bar:
            for row in bar:
                i = bar.pos
                p1, p2 = row.interactions
                label = train_labels[i].view(1, 1)
                if p1 not in prot_idx or p2 not in prot_idx:
                    continue
                pi1, pi2 = prot_idx[p1], prot_idx[p2]
                feat = annots[:, [pi1, pi2]]
                logits = model(g, feat)
                loss = loss_func(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()
        total_loss /= len(train_df)
        model.eval()
        test_loss = 0
        preds = np.zeros(len(test_df), dtype=np.float32)
        with th.no_grad():
            for i, row in enumerate(test_df.itertuples()):
                p1, p2 = row.interactions
                label = test_labels[i].view(1, 1)
                if p1 not in prot_idx or p2 not in prot_idx:
                    continue
                pi1, pi2 = prot_idx[p1], prot_idx[p2]
                feat = annots[:, [pi1, pi2]]
                logits = model(g, feat)
                loss = loss_func(logits, label)
                preds[i] = logits.detach().item()
                test_loss += loss.detach().item()
            test_loss /= len(test_df)
        labels = test_df['labels'].values
        roc_auc = compute_roc(labels, preds)
        print(f'Epoch {epoch}: Loss - {total_loss}, Test loss - {test_loss}, AUC - {roc_auc}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

class PPIModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gcn1 = dglnn.GraphConv(2, 2)
        self.gcn2 = dglnn.GraphConv(2, 2)
        self.gcn3 = dglnn.GraphConv(2, 2)
        self.fc = nn.Linear(572, 1)
        
    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        x = F.relu(x)
        x = th.flatten(x).view(1, -1)
        return th.sigmoid(self.fc(x))
        
def load_ppi_data(train_inter_file, test_inter_file):
    train_df = pd.read_pickle(train_inter_file)
    index = np.arange(len(train_df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = train_df.iloc[index[:10000]]
    
    test_df = pd.read_pickle(test_inter_file)
    index = np.arange(len(test_df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    test_df = test_df.iloc[index[:1000]]
    return train_df, test_df

def load_graph_data(data_file):
    go = Ontology('data/goslim_yeast.obo')
    nodes = list(go.ont.keys())
    node_idx = {v: k for k, v in enumerate(nodes)}
    g = dgl.DGLGraph()
    g.add_nodes(len(nodes))
    for n_id in nodes:
        parents = go.get_parents(n_id)
        parents_idx = [node_idx[go_id] for go_id in parents]
        g.add_edges(node_idx[n_id], parents_idx)
    g = dgl.add_self_loop(g)

    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '559292']
    annotations = np.zeros((len(nodes), len(df)), dtype=np.float32)
    prot_idx = {}
    for i, row in enumerate(df.itertuples()):
        prot_id = row.accessions.split(';')[0]
        prot_idx[prot_id] = i
        for go_id in row.exp_annotations:
            if go_id in node_idx:
                annotations[node_idx[go_id], i] = 1

    return g, annotations, prot_idx
    
if __name__ == '__main__':
    main()

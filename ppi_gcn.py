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
import copy
from torch.utils.data import DataLoader

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
    '--epochs', '-ep', default=32,
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

    train_set_batches = get_batches(g, annots, prot_idx, train_df, train_labels, batch_size)
    test_set_batches = get_batches(g, annots, prot_idx, test_df, test_labels, batch_size)


    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        for iter, (batch, labels) in enumerate(train_set_batches):
            logits = model(batch)

            labels = labels.unsqueeze(1).to(device)
            loss = loss_func(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (iter+1)

        
        model.eval()
        test_loss = 0
        preds = []
        with th.no_grad():

            for iter, (batch, labels) in enumerate(test_set_batches):
                logits = model(batch)
                labels = labels.unsqueeze(1).to(device)
                loss = loss_func(logits, labels)
                test_loss += loss.detach().item()
                preds = np.append(preds, logits.cpu())
            test_loss /= (iter+1)

        labels = test_df['labels'].values
        roc_auc = compute_roc(labels, preds)
        print(f'Epoch {epoch}: Loss - {epoch_loss}, Test loss - {test_loss}, AUC - {roc_auc}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def get_batches(graph, annots, prot_idx, df, labels, batch_size):
    dataset = []
    with ck.progressbar(df.itertuples(), length=len(df)) as bar:
        for row in bar:
            i = bar.pos
            p1, p2 = row.interactions
            label = labels[i].view(1, 1)
            if p1 not in prot_idx or p2 not in prot_idx:
                continue
            pi1, pi2 = prot_idx[p1], prot_idx[p2]
            feat = annots[:, [0, pi1+1, pi2+1]]
            graph_cp = copy.deepcopy(graph)
            graph_cp.ndata['feat'] = feat
            dataset.append((graph_cp, label))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.tensor(labels)

class PPIModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gcn1 = dglnn.GraphConv(3, 3)
        self.gcn2 = dglnn.GraphConv(3, 3)
        self.gcn3 = dglnn.GraphConv(3, 3)
        self.fc = nn.Linear(286*3, 1)
        
    def forward(self, g):
        features = g.ndata['feat']
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        x = F.relu(x)

        x = th.flatten(x).view(-1, 286*3)
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
    go = Ontology('data/goslim_yeast.obo', with_rels=True)
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
    go.calculate_ic(df['prop_annotations'])
    
    annotations = np.zeros((len(nodes), len(df) + 1), dtype=np.float32)
    for i, go_id in enumerate(go.ont.keys()):
        annotations[i, 0] = go.get_ic(go_id)
    prot_idx = {}
    for i, row in enumerate(df.itertuples()):
        prot_id = row.accessions.split(';')[0]
        prot_idx[prot_id] = i
        for go_id in row.prop_annotations:
            if go_id in node_idx:
                annotations[node_idx[go_id], i + 1] = 1

    return g, annotations, prot_idx
    
if __name__ == '__main__':
    main()

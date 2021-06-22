import click as ck
import pandas as pd
from ont import Ontology
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
from dgl.nn.pytorch import RelGraphConv
from baseRGCN import BaseRGCN
from dgl.nn import GraphConv, AvgPooling, MaxPooling




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

    with_ic = False
    if with_ic:
        feat_dim = 3
    else:
        feat_dim = 2


    rels = ['part_of', 'regulates', 'occurs_in']

    g, annots, prot_idx = load_graph_data(data_file, rels = rels, with_ic = with_ic, with_disjoint = True)
    
    num_rels = len(g.canonical_etypes)

 
    g = dgl.to_homogeneous(g)

    num_nodes = g.number_of_nodes()
    print(f"Num nodes: {g.number_of_nodes()}")
    annots = th.FloatTensor(annots)
    train_df, test_df = load_ppi_data(train_inter_file, test_inter_file)
    model = PPIModel(feat_dim, num_rels, num_rels, num_nodes)
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_labels = th.FloatTensor(train_df['labels'].values).to(device)
    test_labels = th.FloatTensor(test_df['labels'].values).to(device)

    train_set_batches = get_batches(g, annots, prot_idx, train_df, train_labels, batch_size, with_ic = with_ic)
    test_set_batches = get_batches(g, annots, prot_idx, test_df, test_labels, batch_size, with_ic = with_ic)


    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        for iter, (batch, labels) in enumerate(train_set_batches):
            logits = model(batch.to(device))

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
                logits = model(batch.to(device))
                labels = labels.unsqueeze(1).to(device)
                loss = loss_func(logits, labels)
                test_loss += loss.detach().item()
                preds = np.append(preds, logits.cpu())
            test_loss /= (iter+1)

        labels = test_df['labels'].values
        roc_auc = compute_roc(labels, preds)
        print(f'Epoch {epoch}: Loss - {epoch_loss}, \tTest loss - {test_loss}, \tAUC - {roc_auc}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def get_batches(graph, annots, prot_idx, df, labels, batch_size, with_ic = False):
    dataset = []
    with ck.progressbar(df.itertuples(), length=len(df)) as bar:
        for row in bar:
            i = bar.pos
            p1, p2 = row.interactions
            label = labels[i].view(1, 1)
            if p1 not in prot_idx or p2 not in prot_idx:
                continue
            pi1, pi2 = prot_idx[p1], prot_idx[p2]
            if with_ic:
                feat = annots[:, [0, pi1+1, pi2+1]]
            else:
                feat = annots[:, [pi1, pi2]]
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


class RGCN(BaseRGCN):

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

class PPIModel(nn.Module):

    def __init__(self, h_dim, num_rels, num_bases, num_nodes):
        super().__init__()
        
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_nodes = num_nodes

        print(f"Num rels: {self.num_rels}")
        print(f"Num bases: {self.num_bases}")

        self.rgcn = RGCN(self.h_dim, 
                        self.h_dim, 
                        self.h_dim, 
                        self.num_rels, 
                        self.num_bases,
                        num_hidden_layers=2, 
                        dropout=0.2,
                        use_self_loop=False, 
                        use_cuda=True
                        )
        
        # self.avgpool = AvgPooling()
        # self.maxpool = MaxPooling()   
        #nn.Linear(4, 1)

        self.fc = nn.Linear(self.num_nodes*self.h_dim, 1) 
        
    def forward(self, g):
        features = g.ndata['feat']
        edge_type = g.edata[dgl.ETYPE].long()

        x = self.rgcn(g, features, edge_type, None)

        #x = th.cat([self.avgpool(g, x), self.maxpool(g, x)], dim=-1)

        x = th.flatten(x).view(-1, self.num_nodes*self.h_dim)
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

def load_graph_data(data_file, rels = [], with_ic = False, with_disjoint = False):
    go = Ontology('data/go.obo', rels, with_disjoint)
    nodes = list(go.ont.keys())
    node_idx = {v: k for k, v in enumerate(nodes)}
   

    g = go.toDGLGraph()
    g = dgl.add_self_loop(g, 'is_a')
    
    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '559292']
    
    if with_ic:
        go.calculate_ic(df['prop_annotations'])
        annotations = np.zeros((len(nodes), len(df) + 1), dtype=np.float32)
        for i, go_id in enumerate(go.ont.keys()):
            annotations[i, 0] = go.get_ic(go_id)

    else:
        annotations = np.zeros((len(nodes), len(df)), dtype=np.float32)

    prot_idx = {}
    for i, row in enumerate(df.itertuples()):
        prot_id = row.accessions.split(';')[0]
        prot_idx[prot_id] = i
        for go_id in row.prop_annotations:
            if go_id in node_idx:
                if with_ic:
                    annotations[node_idx[go_id], i + 1] = 1
                else:
                    annotations[node_idx[go_id], i] = 1
    return g, annotations, prot_idx
    
if __name__ == '__main__':
    main()

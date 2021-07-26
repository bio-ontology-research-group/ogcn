import click as ck
import pandas as pd
from ont import Ontology
import dgl
from dgl import nn as dglnn
import torch as th
import pickle as pkl
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader
from dgl.nn import GraphConv, AvgPooling, MaxPooling
import random

from RelAtt.relGraphConv import RelGraphConv
from RelAtt.baseRGCN import BaseRGCN

from ppi_gcn_rel import GraphDataset

th.manual_seed(0)
np.random.seed(0)
random.seed(0)


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
    '--epochs', '-ep', default=64,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(train_inter_file, test_inter_file, data_file, deepgo_model, model_file, batch_size, epochs, load):
    device = 'cuda'


    with_disjoint = False
    with_intersection = False
    inverse = False

    rels = ['part_of', 'regulates']

    g, annots, prot_idx = load_graph_data(data_file, rels = rels, with_disjoint = with_disjoint, with_intersection = with_intersection, inverse = inverse)
    
    num_rels = len(g.canonical_etypes)
    num_bases = 20
    feat_dim = 2
 
    g = dgl.to_homogeneous(g)

    #print("HOMOGENOUS GRAPH: " + str(g.number_of_edges()))
    
    num_nodes = g.number_of_nodes()
    print(f"Num nodes: {g.number_of_nodes()}")
    annots = th.FloatTensor(annots).to(device)
    train_df, test_df = load_ppi_data(train_inter_file, test_inter_file)
    model = PPIModel(feat_dim, num_rels, num_bases, num_nodes)
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_labels = th.FloatTensor(train_df['labels'].values).to(device)
    test_labels = th.FloatTensor(test_df['labels'].values).to(device)

    train_data = GraphDataset(g, train_df, train_labels, annots, prot_idx)
    test_data = GraphDataset(g, test_df, test_labels, annots, prot_idx)

    train_set_batches = get_batches(train_data, batch_size)
    test_set_batches = get_batches(test_data, batch_size)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        with ck.progressbar(train_set_batches) as bar:
            for iter, (batch_g, batch_feat, batch_labels) in enumerate(bar):
                logits = model(batch_g.to(device), batch_feat)

                labels = batch_labels.unsqueeze(1).to(device)
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
            with ck.progressbar(test_set_batches) as bar:
                for iter, (batch_g, batch_feat, batch_labels) in enumerate(bar):
                    logits = model(batch_g.to(device), batch_feat)
                    labels = batch_labels.unsqueeze(1).to(device)
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

def get_batches(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, features, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.cat(features, dim=0), th.tensor(labels)

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, num_rels, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.n_embedding = th.nn.Linear(num_nodes, h_dim)
        self.e_embedding = th.nn.Embedding(num_rels, h_dim)
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.h_dim = h_dim

    def forward(self, g, hn, r, he, norm):
        return self.n_embedding(hn), self.e_embedding(he.squeeze())

class RGCN(BaseRGCN):

    def build_input_layer(self):
        return EmbeddingLayer(2, self.num_rels, self.h_dim)


    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout, low_mem = True)

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
                        num_hidden_layers=1, 
                        dropout=0.8,
                        use_self_loop=False, 
                        use_cuda=True
                        )
        
        # self.avgpool = AvgPooling()
        # self.maxpool = MaxPooling()   
        # self.fc =  nn.Linear(4, 1)

        self.fc = nn.Linear(self.num_nodes*self.h_dim, 1)
        
    def forward(self, g, features):
        edge_type = g.edata[dgl.ETYPE].long()

        edge_feat = th.arange(self.num_rels).view(-1, 1).long().cuda()

        x, _ = self.rgcn(g, features, edge_type, edge_feat, None)

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

def load_graph_data(data_file, rels = [], with_disjoint = False, with_intersection = False, inverse = True):
    # go = Ontology('data/go.obo', rels, with_disjoint, with_intersection, inverse)
    # nodes = list(go.ont.keys())
    # node_idx = {v: k for k, v in enumerate(nodes)}
   

    # g = go.toDGLGraph()
    
    graphs, data_dict = dgl.load_graphs('data/go_cat.bin')
    g = graphs[0]

    num_nodes = g.number_of_nodes()
    
    with open("data/nodes_cat.pkl", "rb") as pkl_file:
        node_idx = pkl.load(pkl_file)
    g = dgl.add_self_loop(g, 'id')
    
    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '559292']
    
    
    annotations = np.zeros((num_nodes, len(df)), dtype=np.float32)

    prot_idx = {}
    for i, row in enumerate(df.itertuples()):
        prot_id = row.accessions.split(';')[0]
        prot_idx[prot_id] = i
        for go_id in row.prop_annotations:
            if go_id in node_idx:
                annotations[node_idx[go_id], i] = 1
    return g, annotations, prot_idx
    
if __name__ == '__main__':
    main()

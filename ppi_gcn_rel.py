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
from torch.utils.data import DataLoader, IterableDataset
from dgl.nn.pytorch import RelGraphConv
from baseRGCN import BaseRGCN
from dgl.nn import GraphConv, AvgPooling, MaxPooling
import random


import logging
#logging.basicConfig(level=logging.DEBUG)

#th.manual_seed(0)
#np.random.seed(0)
#random.seed(0)


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

    global g, device, num_bases, num_nodes, num_rels, feat_dim, annots, prot_idx, loss_func
    device = 'cuda'
   
    g, annots, prot_idx = load_graph_data(data_file)
    
    num_nodes = g.number_of_nodes()
    print(f"Num nodes: {g.number_of_nodes()}")
    
    annots = th.FloatTensor(annots).to(device)
    num_rels = len(g.canonical_etypes)

    g = dgl.to_homogeneous(g)

    num_bases = 20
    feat_dim = 2
    loss_func = nn.BCELoss()

    
    

    train(batch_size, epochs, data_file, train_inter_file, test_inter_file)
    test(batch_size, data_file, train_inter_file, test_inter_file)
    

def load_data(train_inter_file, test_inter_file):
    train_df, test_df = load_ppi_data(train_inter_file, test_inter_file)
    
    split = int(len(train_df) * 0.8)
    index = np.arange(len(train_df))
    val_df = train_df.iloc[index[split:]]
    train_df = train_df.iloc[index[:split]]

    return train_df, val_df, test_df

def train(batch_size, epochs, data_file, train_inter_file, test_inter_file):

 

    train_df, val_df, _ = load_data(train_inter_file, test_inter_file)

    model = PPIModel(feat_dim, num_rels, num_bases, num_nodes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_labels = th.FloatTensor(train_df['labels'].values).to(device)
    val_labels = th.FloatTensor(val_df['labels'].values).to(device)
    
    train_data = GraphDataset(g, train_df, train_labels, annots, prot_idx)
    val_data = GraphDataset(g, val_df, val_labels, annots, prot_idx)
    
    train_set_batches = get_batches(train_data, batch_size)
    val_set_batches = get_batches(val_data, batch_size)
    
    best_roc_auc = 0
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
        val_loss = 0
        preds = []
        labels = []
        with th.no_grad():
            optimizer.zero_grad()
            with ck.progressbar(val_set_batches) as bar:
                for iter, (batch_g, batch_feat, batch_labels) in enumerate(bar):
                    
                    logits = model(batch_g.to(device), batch_feat)
                    lbls = batch_labels.unsqueeze(1).to(device)
                    loss = loss_func(logits, lbls)
                    val_loss += loss.detach().item()
                    labels = np.append(labels, lbls.cpu())
                    preds = np.append(preds, logits.cpu())
                val_loss /= (iter+1)

        roc_auc = compute_roc(labels, preds)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            th.save(model.state_dict(), 'data/model_rel.pt')
        print(f'Epoch {epoch}: Loss - {epoch_loss}, \tVal loss - {val_loss}, \tAUC - {roc_auc}')

        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune.report(loss=(val_loss), auc=roc_auc)
    print("Finished Training")


def test(batch_size, data_file, train_inter_file, test_inter_file):

    _, _, test_df = load_data(train_inter_file, test_inter_file)
    test_labels = th.FloatTensor(test_df['labels'].values).to(device)
    
    test_data = GraphDataset(g, test_df, test_labels, annots, prot_idx)
    
    test_set_batches = get_batches(test_data, batch_size)

    model = PPIModel(feat_dim, num_rels, num_bases, num_nodes)
    model.load_state_dict(th.load('data/model_rel.pt'))
    model.to(device)
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
    print(f'Test loss - {test_loss}, \tAUC - {roc_auc}')

    return roc_auc

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
                        dropout=0.8,
                        use_self_loop=False, 
                        use_cuda=True
                        )
        
        # self.avgpool = AvgPooling()
        # self.maxpool = MaxPooling()   
        #nn.Linear(4, 1)

        self.fc = nn.Linear(self.num_nodes*self.h_dim, 1) 
        
    def forward(self, g, features):
        edge_type = g.edata[dgl.ETYPE].long()

        x = self.rgcn(g, features, edge_type, None)

        #x = th.cat([self.avgpool(g, x), self.maxpool(g, x)], dim=-1)

        x = th.flatten(x).view(-1, self.num_nodes*self.h_dim)
        return th.sigmoid(self.fc(x))
        


class GraphDataset(IterableDataset):

    def __init__(self, graph, df, labels, annots, prot_idx):
        self.graph = graph
        self.annots = annots
        self.labels = labels
        self.df = df
        self.prot_idx = prot_idx
        
    def get_data(self):
        for i, row in enumerate(self.df.itertuples()):
            p1, p2 = row.interactions
            label = self.labels[i].view(1, 1)
            if p1 not in self.prot_idx or p2 not in self.prot_idx:
                continue
            pi1, pi2 = self.prot_idx[p1], self.prot_idx[p2]
           
            feat = self.annots[:, [pi1, pi2]]

            yield (self.graph, feat, label)

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.df)


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

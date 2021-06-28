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
from torch.utils.data import DataLoader, IterableDataset
from itertools import cycle
import math
import rdflib as rdf

@ck.command()
@ck.option(
    '--hp-file', '-hf', default='data/hp.obo',
    help='Human Phenotype Ontology file in OBO Format')
@ck.option(
    '--data-file', '-df', default='data/data_human.pkl',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--terms-file', '-tf', default='data/hp_terms.csv',
    help='Data file with sequences and complete set of annotations')
@ck.option(
    '--gos-file', '-gf', default='data/go_terms.csv',
    help='DataFrame with list of GO classes (as features)')
@ck.option(
    '--deepgo-model', '-dm', default='data/deepgoplus.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--model-file', '-mf', default='data/pheno.model.h5',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=1,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(hp_file, data_file, terms_file, gos_file, deepgo_model, model_file, batch_size, epochs, load):
    device = 'cuda:0'
    gos_df = pd.read_csv(gos_file)
    gos = gos_df['terms'].values.flatten()
    gos_dict = {v: i for i, v in enumerate(gos)}

    # cross validation settings
    # model_file = f'fold{fold}_exp-' + model_file
    # out_file = f'fold{fold}_exp-' + out_file
    global hpo
    hpo = Ontology(hp_file, with_rels=True)
    terms_df = pd.read_csv(terms_file)
    global terms
    terms = terms_df['terms'].values.flatten()
    print('Phenotypes', len(terms))
    global term_set
    term_set = set(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}
    
    g, annots, labels, train_nids, test_nids = load_graph_data(data_file, gos_dict, terms_dict)
    g = g.to(device)
    annots = th.FloatTensor(annots).to(device)
    labels = th.FloatTensor(labels).to(device)
    
    model = PhenoModel(len(gos), len(terms))
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        with ck.progressbar(train_nids) as bar:
            for nid in bar:
                feat = annots[nid, :].view(-1, 1)
                label = labels[nid, :].view(1, -1)
                logits = model(g, feat)
                
                loss = loss_func(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu()
                
            epoch_loss /= len(train_nids)

        
        model.eval()
        test_loss = 0
        preds = []
        with th.no_grad():
            with ck.progressbar(test_nids) as bar:
                for n_id in bar:
                    feat = annots[n_id, :].view(-1, 1)
                    label = labels[n_id, :].view(1, -1)
                    logits = model(g, feat)
                    loss = loss_func(logits, label)
                    test_loss += loss.detach().cpu()
                    preds = np.append(preds, logits.detach().cpu())
            test_loss /= len(test_nids)

        test_labels = labels[test_nids, :].detach().cpu().numpy()
        roc_auc = compute_roc(test_labels, preds)
        fmax = compute_fmax(test_labels, preds.reshape(len(test_nids), len(terms)))
        print(f'Epoch {epoch}: Loss - {epoch_loss}, Test loss - {test_loss}, AUC - {roc_auc}, Fmax - {fmax}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_fmax(labels, preds):
    fmax = 0.0
    for t in range(1, 50):
        threshold = t / 100.0
        predictions = (preds >= threshold).astype(np.float32)
        tp = np.sum(labels * predictions, axis=1)
        fp = np.sum(predictions, axis=1) - tp
        fn = np.sum(labels, axis=1) - tp
        p = np.mean(tp / (tp + fp))
        r = np.mean(tp / (tp + fn))
        f = 2 * p * r / (p + r)
        if fmax < f:
            fmax = f
            patience = 0
        else:
            patience += 1
            if patience > 10:
                return fmax
    return fmax
        

class MyDataset(IterableDataset):

    def __init__(self, graph, annots, ids, labels):
        self.graph = graph
        self.annots = annots
        self.labels = labels
        self.ids = ids
        
    def get_data(self):
        for nid in self.ids:
            label = self.labels[nid].view(1, -1)
            feat = self.annots[nid, :].view(-1, 1)
            yield (self.graph, feat, label)

    def __iter__(self):
        return self.get_data()
    

def get_batches(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, features, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.cat(features, dim=0), th.cat(labels, dim=0)


class PhenoModel(nn.Module):

    def __init__(self, nb_gos, nb_classes):
        super().__init__()
        self.gcn1 = dglnn.GraphConv(1, 1)
        self.gcn2 = dglnn.GraphConv(1, 1)
        self.gcn3 = dglnn.GraphConv(1, 1)
        self.nb_gos = nb_gos
        self.nb_classes = nb_classes
        self.fc = nn.Linear(nb_gos, nb_classes)
        
    def forward(self, g, features=None):
        if features is None:
            features = g.ndata['feat']
        # x = self.gcn1(g, features)
        # x = self.gcn2(g, x)
        x = features
        x = th.flatten(x).view(-1, self.nb_gos)
        return th.sigmoid(self.fc(x))


def load_graph_data(data_file, gos_dict, terms_dict):
    rg = rdf.Graph().parse(data=open('data/go.owl').read())

    
    g = dgl.DGLGraph()
    g.add_nodes(len(gos_dict))
    for g_id in gos_dict:
        parents = go.get_parents(g_id)
        parents_idx = [gos_dict[go_id] for go_id in parents if go_id in gos_dict]
        g.add_edges(gos_dict[g_id], parents_idx)
    g = dgl.add_self_loop(g)

    df = pd.read_pickle(data_file)
    #Filter out proteins without pheno annotations
    index = []
    for i, row in df.iterrows():
        if len(row.phenotypes) > 0:
            index.append(i)
    df = df.iloc[index]
    
    index = np.arange(len(df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(len(df) * .9)
    train_nids = index[:train_n]
    test_nids = index[train_n:]
    
    annotations = np.zeros((len(df), len(gos_dict)), dtype=np.float32)
    labels = np.zeros((len(df), len(terms_dict)), dtype=np.float32)
    
    for i, row in enumerate(df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in gos_dict:
                annotations[i, gos_dict[go_id]] = 1
        for hp_id in row.prop_phenotypes:
            if hp_id in terms_dict:
                labels[i, terms_dict[hp_id]] = 1

    return g, annotations, labels, train_nids, test_nids
    
if __name__ == '__main__':
    main()

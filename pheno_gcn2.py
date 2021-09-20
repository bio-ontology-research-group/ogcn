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
    '--model-file', '-mf', default='data/phenogcn2.th',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=12,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=64,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(hp_file, data_file, terms_file, gos_file, deepgo_model, model_file, batch_size, epochs, load):
    device = 'cuda:1'
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
    hpo_matrix = get_hpo_matrix(hpo, terms_dict)
    hpo_matrix = th.FloatTensor(hpo_matrix).to(device)
    g, etypes, annots, labels, train_nids, valid_nids, test_nids = load_graph_data(data_file, gos_dict, terms_dict)
    g = g.to(device)
    etypes = etypes.to(device)
    annots = th.FloatTensor(annots).to(device)
    labels = th.FloatTensor(labels).to(device)

    train_dataset = MyDataset(g, etypes, annots, train_nids, labels)
    train_batches, train_steps = get_batches(train_dataset, batch_size)
    
    model = PhenoModel(len(gos), len(terms))
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_loss = 1000000.0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        with ck.progressbar(train_batches) as bar:
            for batch_g, batch_etypes, batch_feat, batch_labels in bar:
                # feat = annots[nid, :].view(-1, 1)
                # label = labels[nid, :].view(1, -1)
                logits = model(batch_g, batch_etypes, batch_feat)
                
                loss = loss_func(logits, batch_labels)
                # hierarchical loss
                loss += th.sum(th.relu((logits.unsqueeze(2) - logits.unsqueeze(1)) * hpo_matrix))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu()
                
            epoch_loss /= train_steps

        
        model.eval()
        valid_loss = 0
        preds = []
        with th.no_grad():
            with ck.progressbar(valid_nids) as bar:
                for n_id in bar:
                    feat = annots[n_id, :].view(-1, 1)
                    label = labels[n_id, :].view(1, -1)
                    logits = model(g, etypes, feat)
                    loss = loss_func(logits, label)
                    valid_loss += loss.detach().cpu()
                    preds = np.append(preds, logits.detach().cpu())
                valid_loss /= len(valid_nids)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model')
                    th.save(model.state_dict(), model_file)


        valid_labels = labels[valid_nids, :].detach().cpu().numpy()
        roc_auc = compute_roc(valid_labels, preds)
        fmax = compute_fmax(valid_labels, preds.reshape(len(valid_nids), len(terms)))
        print(f'Epoch {epoch}: Loss - {epoch_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}, Fmax - {fmax}')

    # Loading best model
    print('Loading the best model')
    model.load_state_dict(th.load(model_file))
    model.eval()

    test_loss = 0
    preds = []
    with th.no_grad():
        with ck.progressbar(test_nids) as bar:
            for n_id in bar:
                feat = annots[n_id, :].view(-1, 1)
                label = labels[n_id, :].view(1, -1)
                logits = model(g, etypes, feat)
                loss = loss_func(logits, label)
                test_loss += loss.detach().cpu()
                preds = np.append(preds, logits.detach().cpu())
            test_loss /= len(test_nids)

        test_labels = labels[test_nids, :].detach().cpu().numpy()
        roc_auc = compute_roc(test_labels, preds)
        fmax = compute_fmax(test_labels, preds.reshape(len(test_nids), len(terms)))
        print(f'Test loss - {test_loss}, AUC - {roc_auc}, Fmax - {fmax}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def compute_fmax(labels, preds):
    fmax = 0.0
    patience = 0
    for t in range(1, 101):
        threshold = t / 100.0
        predictions = (preds >= threshold).astype(np.float32)
        tp = np.sum(labels * predictions, axis=1)
        fp = np.sum(predictions, axis=1) - tp
        fn = np.sum(labels, axis=1) - tp
        tp_ind = tp > 0
        tp = tp[tp_ind]
        fp = fp[tp_ind]
        fn = fn[tp_ind]
        if len(tp) == 0:
            continue
        p = np.mean(tp / (tp + fp))
        r = np.sum(tp / (tp + fn)) / len(tp_ind)
        f = 2 * p * r / (p + r)
        if fmax <= f:
            fmax = f
    return fmax
        

class MyDataset(IterableDataset):

    def __init__(self, graph, etypes, annots, ids, labels):
        self.graph = graph
        self.etypes = etypes
        self.annots = annots
        self.labels = labels
        self.ids = ids
        
    def get_data(self):
        for nid in self.ids:
            label = self.labels[nid].view(1, -1)
            feat = self.annots[nid, :].view(-1, 1)
            yield (self.graph, self.etypes, feat, label)

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return len(self.ids)

def get_batches(dataset, batch_size):
    steps = int(math.ceil(len(dataset) / batch_size))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return data_loader, steps

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, etypes, features, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.cat(etypes, dim=0), th.cat(features, dim=0), th.cat(labels, dim=0)


def get_hpo_matrix(hpo, terms_dict):
    nb_classes = len(terms_dict)
    res = np.zeros((nb_classes, nb_classes), dtype=np.float32)
    for hp_id, i in terms_dict.items():
        subs = hpo.get_term_set(hp_id)
        res[i, i] = 1
        for h_id in subs:
            if h_id in terms_dict:
                res[terms_dict[h_id], i] = 1
    return res


class PhenoModel(nn.Module):

    def __init__(self, nb_gos, nb_classes):
        super().__init__()
        self.gcn1 = dglnn.RelGraphConv(1, 1, 11)
        self.gcn2 = dglnn.RelGraphConv(1, 1, 11)
        # self.gcn3 = dglnn.RelGraphConv(1, 1, 11)
        self.nb_gos = nb_gos
        self.nb_classes = nb_classes
        self.fc1 = nn.Linear(nb_gos, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        # self.out = nn.Linear(nb_gos, nb_classes)
        self.out = nn.Linear(1000, nb_classes)
        
    def forward(self, g, etypes, features=None):
        # if features is None:
        #     features = g.ndata['feat']
        # x = self.gcn1(g, features, etypes)
        x = features
        x = th.flatten(x).view(-1, self.nb_gos)
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        return th.sigmoid(self.out(x))


def load_graph_data(data_file, gos_dict, terms_dict):
    graphs, data_dict = dgl.load_graphs('data/go.bin')
    g = graphs[0]
    etypes = data_dict['etypes']
    # g = dgl.add_self_loop(g)

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
    valid_n = int(train_n * .9)
    train_nids = index[:valid_n]
    valid_nids = index[valid_n:train_n]
    test_nids = index[train_n:]

    print(f'Train: {len(train_nids)}, Valid: {len(valid_nids)}, Test: {len(test_nids)}')
    
    annotations = np.zeros((len(df), len(gos_dict)), dtype=np.float32)
    labels = np.zeros((len(df), len(terms_dict)), dtype=np.float32)
    
    for i, row in enumerate(df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in gos_dict:
                annotations[i, gos_dict[go_id]] = 1
        # for p_id in row.interactions:
        #     annotations[i, gos_dict[p_id]] = 1
        for hp_id in row.prop_phenotypes:
            if hp_id in terms_dict:
                labels[i, terms_dict[hp_id]] = 1

    return g, etypes, annotations, labels, train_nids, valid_nids, test_nids
    
if __name__ == '__main__':
    main()

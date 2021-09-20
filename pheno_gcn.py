#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import logging
import math
import time
from collections import deque
import torch as th
import dgl
from dgl.nn import GraphConv, RelGraphConv
import torch.nn.functional as F
from torch import nn
import os

from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from aminoacids import MAXLEN, to_onehot
from utils import Ontology, FUNC_DICT, is_exp_code

logging.basicConfig(level=logging.DEBUG)

print("GPU Available: ", th.cuda.is_available())


class HPOLayer(object):

    def __init__(self, nb_classes, **kwargs):
        self.nb_classes = nb_classes
        self.hpo_matrix = np.zeros((nb_classes, nb_classes), dtype=np.float32)
        super(HPOLayer, self).__init__(**kwargs)

    def set_hpo_matrix(self, hpo_matrix):
        self.hpo_matrix = hpo_matrix

    def get_config(self):
        config = super(HPOLayer, self).get_config()
        config['nb_classes'] = self.nb_classes
        return config
    
    def build(self, input_shape):
        self.kernel = K.variable(
            self.hpo_matrix, name='{}_kernel'.format(self.name))
        self.non_trainable_weights.append(self.kernel)
        super(HPOLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = tf.keras.backend.repeat(x, self.nb_classes)
        return tf.math.multiply(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.nb_classes, self.nb_classes] 


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
    '--model-file', '-mf', default='data/model.th',
    help='DeepGOPlus model')
@ck.option(
    '--out-file', '-o', default='data/predictions.pkl',
    help='Result file with predictions for test set')
@ck.option(
    '--fold', '-f', default=1,
    help='Fold index')
@ck.option(
    '--batch-size', '-bs', default=5,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=24,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--logger-file', '-lf', default='data/training.csv',
    help='Batch size')
@ck.option(
    '--threshold', '-th', default=0.5,
    help='Prediction threshold')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Prediction threshold')
def main(hp_file, data_file, terms_file, gos_file, model_file,
         out_file, fold, batch_size, epochs, load, logger_file, threshold,
         device):
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
    print('Loading data')
    ppi_g, go_g, go_etypes, features, labels, train_nids, test_nids, test_df = load_data(data_file, terms_dict, gos_dict, fold)
    net = Net(len(gos), len(terms)).to(device)
    # g, features, labels, train_nids, test_nids = g.to(device), features.to(device), labels.to(device), train_nids.to(device), test_nids.to(device)
    print(ppi_g, go_g)
    labels = labels.to(device)
    features = features.to(device)
    go_etypes = go_etypes.to(device)
    print(net)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.NodeDataLoader(
        ppi_g, train_nids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        device=device
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        ppi_g, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        device=device
    )
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_nids)/batch_size))
            with ck.progressbar(length=train_steps) as bar:
                for input_nodes, output_nodes, blocks in dataloader:
                    bar.update(1)
                    batch_go_g = dgl.batch([go_g] * len(input_nodes)).to(device)
                    batch_etypes = th.cat([go_etypes] * len(input_nodes)).to(device)
                    logits = net(batch_go_g, batch_etypes, blocks, features[input_nodes])
                    loss = F.binary_cross_entropy(logits, labels[output_nodes])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
                    
            train_loss /= train_steps
            net.eval()
            with th.no_grad():
                _loss = 0
                test_steps = int(math.ceil(len(test_nids) / batch_size))
                test_loss = 0
                preds = []
                for input_nodes, output_nodes, blocks in test_dataloader:
                    batch_go_g = dgl.batch([go_g] * len(input_nodes)).to(device)
                    batch_etypes = th.cat([go_etypes] * len(input_nodes)).to(device)
                    logits = net(batch_go_g, batch_etypes, blocks, features[input_nodes])
                    batch_loss = F.binary_cross_entropy(logits, labels[output_nodes])
                    test_loss += batch_loss.detach().item()
                    preds = np.append(preds, logits.detach().cpu().numpy())
                test_loss /= test_steps
                valid_labels = labels[test_nids].detach().cpu().numpy()
                roc_auc = compute_roc(valid_labels, preds)
                fmax = compute_fmax(valid_labels, preds.reshape(len(test_nids), len(terms)))
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {test_loss}, AUC - {roc_auc}, Fmax - {fmax}')

                if test_loss < best_loss:
                    best_loss = test_loss
                    print('Saving model')
                    th.save(net.state_dict(), model_file)

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_nids) / batch_size))
        test_loss = 0
        preds = []
        for input_nodes, output_nodes, blocks in test_dataloader:
            batch_go_g = dgl.batch([go_g] * len(input_nodes)).to(device)
            batch_etypes = th.cat([go_etypes] * len(input_nodes)).to(device)
            logits = net(batch_go_g, batch_etypes, blocks, features[input_nodes])
            batch_loss = F.binary_cross_entropy(logits, labels[output_nodes])
            test_loss += batch_loss.detach().item()
            preds = np.append(preds, logits.detach().cpu().numpy())
        test_loss /= test_steps
        preds = preds.reshape(len(test_nids), len(terms))
        valid_labels = labels[test_nids].detach().cpu().numpy()
        roc_auc = compute_roc(valid_labels, preds)
        fmax = compute_fmax(valid_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}, Fmax - {fmax}')

    test_df['preds'] = list(preds)

    test_df.to_pickle(out_file)
    
        
class Net(nn.Module):

    def __init__(self, input_length, nb_classes, hidden_dim=1024):
        super().__init__()
        self.rgcn0 = RelGraphConv(1, 1, 11)
        self.fc0 = nn.Linear(input_length, hidden_dim)
        self.gcn1 = GraphConv(hidden_dim, hidden_dim)
        # self.gcn2 = GraphConv(hidden_dim, nb_classes)
        self.fc1 = nn.Linear(hidden_dim, nb_classes)
        self.dropout = nn.Dropout()
        
    def forward(self, go_g, go_etypes, blocks, x):
        n = x.shape[0]
        n_gos = x.shape[1]
        x = th.transpose(x, 0, 1).reshape(n * n_gos, 1)
        x = self.rgcn0(go_g, x, go_etypes)
        x = x.view(-1, n)
        x = th.transpose(x, 0, 1)
        x = self.dropout(th.relu(self.fc0(x)))
        x = self.gcn1(blocks[0], x)
        # x = self.gcn2(blocks[1], x)
        x = th.sigmoid(self.fc1(x))
        return x


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


def load_data(data_file, terms_dict, gos_dict, fold=1):
    from dgl import save_graphs, load_graphs
    df = pd.read_pickle(data_file)
    n = len(df)
    graphs, data_dict = dgl.load_graphs('data/ppi.bin')
    ppi_g = graphs[0]
    graphs, data_dict = dgl.load_graphs('data/go.bin')
    go_g = graphs[0]
    go_etypes = data_dict['etypes']
    
    proteins = df['proteins']
    prot_idx = {v: k for k, v in enumerate(proteins)}
    features = np.zeros((n, len(gos_dict)), dtype=np.float32)
    labels = np.zeros((n, len(terms_dict)), dtype=np.float32)
    # Filter proteins with annotations
    for i, row in enumerate(df.itertuples()):
        # for go_id, score in row.dg_annotations.items():
        #     if go_id in gos_dict:
        #         features[i, gos_dict[go_id]] = score
        for go_id in row.prop_annotations:
            if go_id in gos_dict:
                features[i, gos_dict[go_id]] = 1
        if len(row.phenotypes) > 0:
            for hp_id in row.prop_phenotypes:
                if hp_id in terms_dict:
                    labels[i, terms_dict[hp_id]] = 1
    features = th.FloatTensor(features)
    labels = th.FloatTensor(labels)
    ppi_g = dgl.add_self_loop(ppi_g)


    index = []
    for i, row in enumerate(df.itertuples()):
        if len(row.phenotypes) > 0:
            index.append(i)
    
    index = np.array(index)
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_n = int(len(index) * 0.9)
    train_index = index[:train_n]
    test_index = index[train_n:]

    test_df = df.iloc[test_index]
    
    train_index = th.LongTensor(train_index)
    test_index = th.LongTensor(test_index)
    
    
    return ppi_g, go_g, go_etypes, features, labels, train_index, test_index, test_df

if __name__ == '__main__':
    main()

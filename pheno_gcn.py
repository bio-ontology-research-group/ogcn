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
from dgl.nn import GraphConv
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
    '--batch-size', '-bs', default=32,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=1024,
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
    '--device', '-d', default='cpu:0',
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
    g, features, labels, train_nids, test_nids, test_df = load_data(data_file, terms_dict, gos_dict, fold)
    g, features, labels, train_nids, test_nids = g.to(device), features.to(device), labels.to(device), train_nids.to(device), test_nids.to(device)
    net = Net(len(gos), len(terms)).to(device)
    print(net)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g, test_nids, sampler,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )
    optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            for input_nodes, output_nodes, blocks in dataloader:
                logits = net(blocks, features[input_nodes])
                loss = F.binary_cross_entropy(logits, labels[output_nodes])
                print('Train loss', loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            net.eval()
            with th.no_grad():
                test_loss = 0
                for input_nodes, output_nodes, blocks in test_dataloader:
                    logits = net(blocks, features[input_nodes])
                    batch_loss = F.binary_cross_entropy(logits, labels[output_nodes])
                    test_loss += batch_loss.item()
                print(f"Epoch {epoch} | Loss {loss.item():.4f} | Test Loss {test_loss:.4f}")
                if test_loss < best_loss:
                    best_loss = test_loss
                    print('Saving model')
                    th.save(net.state_dict(), model_file)

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    predictions = net(g, features)[test_mask]
    labels = labels[test_mask]

    roc_auc = compute_roc(
        labels.cpu().detach().numpy(), predictions.cpu().detach().numpy())
    print('ROC AUC', roc_auc)
    predictions = predictions.cpu().detach().numpy()
    print(predictions.shape)
    test_df['preds'] = list(predictions)

    test_df.to_pickle(out_file)
    
        
class Net(nn.Module):

    def __init__(self, input_length, nb_classes):
        super().__init__()
        self.gcn1 = GraphConv(input_length, nb_classes)
        # self.gcn2 = GraphConv(1000, nb_classes)
        # self.fc1 = nn.Linear(input_length, nb_classes)

    def forward(self, blocks, x):
        x = self.gcn1(blocks[0], x)
        # x = self.gcn2(blocks[1], x)
        x = F.sigmoid(x)
        return x


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def load_data(data_file, terms_dict, gos_dict, fold=1):
    from dgl import save_graphs, load_graphs
    df = pd.read_pickle(data_file)
    n = len(df)
    graph_path = data_file + '.bin'
    if not os.path.exists(graph_path):
        # Build the PPI graph
        g = dgl.DGLGraph()
        g.add_nodes(n)
        proteins = df['proteins']
        prot_idx = {v: k for k, v in enumerate(proteins)}
        features = np.zeros((n, len(gos_dict)), dtype=np.float32)
        labels = np.zeros((n, len(terms_dict)), dtype=np.float32)
        # Filter proteins with annotations
        for i, row in enumerate(df.itertuples()):
            edges = [prot_idx[p_id] for p_id in row.interactions]
            if len(edges) > 0:
                g.add_edges(i, edges)
            for go_id in row.prop_annotations:
                if go_id in gos_dict:
                    features[i, gos_dict[go_id]] = 1
            if len(row.phenotypes) > 0:
                for hp_id in row.prop_phenotypes:
                    if hp_id in terms_dict:
                        labels[i, terms_dict[hp_id]] = 1
        features = th.FloatTensor(features)
        labels = th.FloatTensor(labels)
        g = dgl.add_self_loop(g)
        save_graphs(graph_path, g, {'features': features, 'labels': labels})
    else:
        gs, data_dict = load_graphs(graph_path)
        print(gs)
        g = gs[0]
        features, labels = data_dict['features'], data_dict['labels']

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
    train_mask = np.zeros(n, dtype=np.bool)
    train_mask[train_index] = True

    test_mask = np.zeros(n, dtype=np.bool)
    test_mask[test_index] = True

    train_index = th.LongTensor(train_index)
    test_index = th.LongTensor(test_index)
    
    test_df = df.iloc[test_index]
    
    return g, features, labels, train_index, test_index, test_df

if __name__ == '__main__':
    main()

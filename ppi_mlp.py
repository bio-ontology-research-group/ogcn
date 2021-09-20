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
from sagpool import SAGNetworkHierarchical
from dgl.nn import GraphConv, AvgPooling, MaxPooling
import random

th.manual_seed(0)
np.random.seed(0)
random.seed(0)

ORG_ID = '4932'

@ck.command()
@ck.option(
    '--train-inter-file', '-trif', default=f'data/{ORG_ID}.train_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--valid-inter-file', '-tsif', default=f'data/{ORG_ID}.valid_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--test-inter-file', '-tsif', default=f'data/{ORG_ID}.test_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--data-file', '-df', default='data/swissprot_2021_03.pkl',
    help='Data file with protein sequences')
@ck.option(
    '--model-file', '-mf', default=f'data/{ORG_ID}.mlp.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')


def main(train_inter_file, valid_inter_file, test_inter_file, data_file,  model_file, batch_size, epochs, load):
    device = 'cuda'
    train, valid, test, annots = load_data(
        data_file, train_inter_file, valid_inter_file, test_inter_file)
    model = PPIModel()
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
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
        print(f'Epoch {epoch}: Loss - {epoch_loss}, Test loss - {test_loss}, AUC - {roc_auc}')

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


class PPIModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        
    def forward(self, g):
        features = g.ndata['feat']
        x = self.gcn1(g, features)
        x = F.relu(x)
        x = self.gcn2(g, x)
        x = F.relu(x)

        x = th.cat([self.avgpool(g, x), self.maxpool(g, x)], dim=-1)
        #x = th.flatten(x).view(-1, self.num_nodes*2)
        return th.sigmoid(self.fc(x))
        
def load_ppi_data(train_inter_file, test_inter_file):
    train_df = pd.read_pickle(train_inter_file)
    index = np.arange(len(train_df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = train_df.iloc[index[:1000]]
    
    test_df = pd.read_pickle(test_inter_file)
    index = np.arange(len(test_df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    test_df = test_df.iloc[index[:1000]]
    return train_df, test_df


if __name__ == '__main__':
    main()

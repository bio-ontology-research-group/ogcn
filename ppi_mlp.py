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
import random
from torch_utils import FastTensorDataLoader
import math

th.manual_seed(0)
np.random.seed(0)
random.seed(0)

ORG_ID = '9606'

@ck.command()
@ck.option(
    '--train-inter-file', '-trif', default=f'data/{ORG_ID}.train_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--valid-inter-file', '-vlif', default=f'data/{ORG_ID}.valid_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--test-inter-file', '-tsif', default=f'data/{ORG_ID}.test_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--terms-file', '-tf', default=f'data/{ORG_ID}.terms.tsv',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--data-file', '-df', default='data/swissprot_2021_03.pkl',
    help='Data file with protein sequences')
@ck.option(
    '--model-file', '-mf', default=f'data/{ORG_ID}.mlp.h5',
    help='Prediction model')
@ck.option(
    '--out-file', '-of', default=f'data/{ORG_ID}.mlp_scores.tsv',
    help='Prediction results')
@ck.option(
    '--batch-size', '-bs', default=128,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(train_inter_file, valid_inter_file, test_inter_file, terms_file, data_file,  model_file, out_file, batch_size, epochs, load):
    device = 'cuda'

    with open(terms_file) as f:
        terms = f.read().splitlines()
    terms_dict = {v:k for k, v in enumerate(terms)}

    annots, train, valid, test, test_df = load_data(
        data_file, terms_dict, train_inter_file, valid_inter_file, test_inter_file)
    annots = annots.to(device)

    train_data, train_labels = train
    valid_data, valid_labels = valid
    test_data, test_labels = test

    print(len(train_data), len(valid_data), len(test_data))
    valid_labels = valid_labels.detach().numpy()
    test_labels = test_labels.detach().numpy()
    
    model = PPIModel(len(terms_dict))
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    train_loader = FastTensorDataLoader(*train, batch_size=batch_size, shuffle=False)
    valid_loader = FastTensorDataLoader(*valid, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(*test, batch_size=batch_size, shuffle=False)

    if not load:
        best_loss = 100000
        for epoch in range(epochs):
            epoch_loss = 0
            model.train()
            train_steps = int(math.ceil(len(train[0]) / batch_size))

            with ck.progressbar(train_loader, show_pos=True) as bar:
                for batch_data, batch_labels in bar:
                    annots_p1 = annots[batch_data[:, 0].to(device)]
                    annots_p2 = annots[batch_data[:, 1].to(device)]
                    logits = model(annots_p1, annots_p2)
                    loss = loss_func(logits, batch_labels.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                epoch_loss /= train_steps


            model.eval()
            valid_loss = 0
            valid_steps = int(math.ceil(len(valid[0]) / batch_size))
            preds = []
            with th.no_grad():
                with ck.progressbar(valid_loader, show_pos=True) as bar:
                    for batch_data, batch_labels in bar:
                        annots_p1 = annots[batch_data[:, 0].to(device)]
                        annots_p2 = annots[batch_data[:, 1].to(device)]
                        logits = model(annots_p1, annots_p2)
                        loss = loss_func(logits, batch_labels.to(device))
                        valid_loss += loss.detach().item()
                        preds = np.append(preds, logits.cpu())
                valid_loss /= valid_steps
            roc_auc = compute_roc(valid_labels, preds)
            print(f'Epoch {epoch}: Loss - {epoch_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(model.state_dict(), model_file)

    print('Loading the best model')
    model.load_state_dict(th.load(model_file))
    model.eval()
    test_loss = 0
    test_steps = int(math.ceil(len(test_labels) / batch_size))
    preds = []
    with th.no_grad():
        with ck.progressbar(test_loader, show_pos=True) as bar:
            for batch_data, batch_labels in bar:
                annots_p1 = annots[batch_data[:, 0].to(device)]
                annots_p2 = annots[batch_data[:, 1].to(device)]
                logits = model(annots_p1, annots_p2)
                loss = loss_func(logits, batch_labels.to(device))
                test_loss += loss.detach().item()
                preds = np.append(preds, logits.cpu())
        test_loss /= test_steps
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test loss - {test_loss}, AUC - {roc_auc}')

    with open(out_file, 'w') as f:
        for i, row in enumerate(test_df.itertuples()):
            p1, p2 = row.interactions
            score = preds[i]
            f.write(f'{p1}\t{p2}\t{score}\n')


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


class PPIModel(nn.Module):

    def __init__(self, annots_length, hidden_dim=1024):
        super().__init__()
        self.annots_length = annots_length
        self.fc = nn.Linear(annots_length, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, annots_p1, annots_p2):
        x1 = self.dropout(self.fc(annots_p1))
        x2 = self.dropout(self.fc(annots_p2))
        x = th.sum(x1 * x2, dim=1, keepdims=True)
        return th.sigmoid(x)

def get_data(df, prot_idx):
    data = []
    labels = []
    for i, row in enumerate(df.itertuples()):
        p1, p2 = row.interactions
        data.append((prot_idx[p1], prot_idx[p2]))
        labels.append(row.labels)
    data = th.LongTensor(data)
    labels = th.FloatTensor(labels).view(-1, 1)
    return data, labels


def load_data(data_file, terms_dict, train_inter_file, valid_inter_file, test_inter_file):
    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '9606']
    
    prot_idx = {}
    annots = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        p_id = row.accessions.split(';')[0].strip()
        prot_idx[p_id] = i
        for go_id in row.prop_annotations:
            annots[i, terms_dict[go_id]] = 1
    
    train_df = pd.read_pickle(train_inter_file)
    valid_df = pd.read_pickle(valid_inter_file)
    test_df = pd.read_pickle(test_inter_file)
    
    index = np.arange(len(train_df))
    np.random.seed(seed=0)
    np.random.shuffle(index)
    train_df = train_df.iloc[index]
    
    train = get_data(train_df, prot_idx)
    valid = get_data(valid_df, prot_idx)
    test = get_data(test_df, prot_idx)
    return annots, train, valid, test, test_df


if __name__ == '__main__':
    main()

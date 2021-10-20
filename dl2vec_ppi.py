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
from torch.autograd import Variable
import pickle as pkl

th.manual_seed(0)
np.random.seed(0)
random.seed(0)

ORG_ID = 'yeast'

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
    '--model-file', '-mf', default=f'../{ORG_ID}.mlp.h5',
    help='Prediction model')
@ck.option(
    '--out-file', '-of', default=f'../{ORG_ID}.mlp_scores.tsv',
    help='Prediction results')
@ck.option(
    '--batch-size', '-bs', default=128,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--inter-file', '-if', default='', help='Test file')
    
def main(train_inter_file, valid_inter_file, test_inter_file,  model_file, out_file, batch_size, epochs, load, inter_file ):
    device = 'cuda'
    train = pd.read_pickle(train_inter_file)  
    valid = pd.read_pickle(valid_inter_file)  
    test = pd.read_pickle(test_inter_file)  
    
    train = convert_datalodaer(train)
    valid = convert_datalodaer(valid)
    test = convert_datalodaer(test)        
    
    model = PPIModel(100)
    model.to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    
    if not load:
        best_loss = 100000
        best_auc=0
        i=0
        for epoch in range(epochs):
            epoch_loss = 0
            model.train()
            train_steps = int(math.ceil(len(train) / batch_size))

            with ck.progressbar(train_loader, show_pos=True) as bar:
                for batch_id, (x) in enumerate(train_loader):
                    annots_p1 = Variable(x['p1'].cuda()) 
                    annots_p2 = Variable(x['p2'].cuda())
                    label = Variable(x['label'].cuda()) 

                    logits = model(annots_p1, annots_p2)
                    label = label.view(-1,1,1)
                    loss = loss_func(logits, label.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()

                epoch_loss /= train_steps
                #epoch_loss /= len(train_loader)

          
            model.eval()
            valid_loss = 0
            valid_steps = int(math.ceil(len(valid) / batch_size))
            preds = []
            trues = []
            with th.no_grad():
                with ck.progressbar(valid_loader, show_pos=True) as bar:
                    for batch_id, (x) in enumerate(valid_loader):
                        annots_p1 = Variable(x['p1'].cuda()) 
                        annots_p2 = Variable(x['p1'].cuda())
                        label = Variable(x['label'].cuda()) 
                        label = label.view(-1,1,1)   
                        logits = model(annots_p1, annots_p2)
                        loss = loss_func(logits, label.to(device))
                        valid_loss += loss.detach().item()
                        
                        preds = np.append(preds, logits.data.cpu().numpy())
                        trues = np.append(trues, label.data.cpu().numpy())

                valid_loss /= valid_steps
                #valid_loss /=  len(valid_loader)
                            
            roc_auc = compute_roc(trues, preds)
            
            print(f'Epoch {epoch}: Loss - {epoch_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(model.state_dict(), model_file)
            else:
                i+=1

            if i >= 20:
                break
            #if roc_auc > best_auc:
            #    best_auc = roc_auc
            #    print('Saving model')
            #    th.save(model.state_dict(), model_file)


    print('Loading the best model')
    model.load_state_dict(th.load(model_file))
    model.eval()
    test_loss = 0
    test_steps = int(math.ceil(len(test) / batch_size))
    preds = []
    trues = []
    with th.no_grad():
        with ck.progressbar(test_loader, show_pos=True) as bar:
            for batch_id, (x) in enumerate(test_loader):
                annots_p1 = Variable(x['p1'].cuda()) 
                annots_p2 = Variable(x['p1'].cuda()) 
                label = Variable(x['label'].cuda())
                label = label.view(-1,1,1)   
                
                logits = model(annots_p1, annots_p2)
                loss = loss_func(logits, label.to(device))
                test_loss += loss.detach().item()
                preds = np.append(preds, logits.cpu())
                trues = np.append(trues, label.data.cpu().numpy())
                
        test_loss /= test_steps
        #test_loss /=  len(test_loader)
        roc_auc = compute_roc(trues, preds)
        
        print(f'Test loss - {test_loss}, AUC - {roc_auc}')

    with open(out_file, 'w') as f:        
        x = pd.read_csv(inter_file, sep='\t')
        for i, row in enumerate(x.itertuples()):
            p= row.interactions
            score = preds[i]
            f.write(f'{p}\t{score}\n')


def convert_datalodaer(dic):
    data=[]
    for k,v in dic.items():
        embed_dic={}
        embed_dic['p1'] = th.tensor(v['p1']).reshape(1,100)
        embed_dic['p2'] = th.tensor(v['p2']).reshape(1,100)
        embed_dic['label'] = th.tensor(float(v['label']))
        data.append(embed_dic)
        
    return data
    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return roc_auc

class PPIModel(nn.Module):

    def __init__(self, annots_length, hidden_dim=256):
        super().__init__()
        self.annots_length = annots_length
        self.fc = nn.Linear(annots_length, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.6)
        
    def forward(self, annots_p1, annots_p2):
        x1 = th.relu(self.fc(annots_p1))
        x1 = self.dropout(x1)
        
        x2 = th.relu(self.fc(annots_p2))
        x2 = self.dropout(x2)
        
        x = th.sum(x1 * x2, dim=1, keepdims=True)
        return th.sigmoid(self.out(x))

    '''
    def __init__(self, annots_length, hidden_dim=256):
        super().__init__()
        self.annots_length = annots_length
        self.fc = nn.Linear(annots_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 40)
        self.out = nn.Linear(40, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, annots_p1, annots_p2):
        x1 = th.relu(self.fc(annots_p1))
        x1 = self.dropout(x1)
        x1 = th.relu(self.fc2(x1))
        
        x2 = th.relu(self.fc(annots_p2))
        x2 = self.dropout(x2)
        x2 = th.relu(self.fc2(x2))        
        
        x = th.sum(x1 * x2, dim=1, keepdims=True)
        return th.sigmoid(self.out(x))
    '''
    
if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial

import logging
#logging.basicConfig(level=logging.DEBUG)

class RelAttLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RelAttLayer, self).__init__()
        
        self.shared_weight = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_dim,
                                                self.out_dim))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))

        # init trainable parameters

        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.shared_weight.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

        nn.init.xavier_uniform_(self.weight, gain=gain)
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp, gain=gain)
        if self.bias:
            nn.init.xavier_uniform_(self.bias,gain=gain)


    def edge_attention(self, edges):

        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_dim, self.num_bases, self.out_dim)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_dim, self.out_dim)
        else:
            weight = self.weight

        if self.is_input_layer:
            embed = weight.view(-1, self.out_dim)
            index_src = edges.data['rel_type'] * self.in_dim + edges.src['id']
            index_dst = edges.data['rel_type'] * self.in_dim + edges.dst['id']
             
            h_src = embed[index_src]
            z_src = self.shared_weight(h_src)
            h_dst = embed[index_dst]
            z_dst = self.shared_weight(h_dst)
            he = edges.data['feat']
        else:
            h_src = edges.src['h']
            z_src = self.shared_weight(h_src)
            h_dst = edges.dst['h']
            z_dst = self.shared_weight(h_dst)
            he = edges.data['h']
        logging.debug("shared weight dims: " + str(self.in_dim) + ", " + str(self.out_dim))
        logging.debug("embeddings dims: " + str(he.shape))

        ze = self.shared_weight(he)


        
        edges.src['z'] = z_src
        edges.dst['z'] = z_dst
        edges.data['z'] = ze
        edges.data['h'] = ze

        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.data['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}
        

    def forward(self, g):

        logging.debug("In layer forward")
        


        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_dim, self.num_bases, self.out_dim)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_dim, self.out_dim)
        else:
            weight = self.weight

        
        g.apply_edges(self.edge_attention)

        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_dim)

                index = edges.data['rel_type'] * self.in_dim + edges.src['id']

                logging.debug("rel types " + str(edges.data['rel_type'].shape))
                return {'msg': embed[index] * edges.data['e']}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['e']
                return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)




###############################################################################
# Full R-GCN model defined
# ~~~~~~~~~~~~~~~~~~~~~~~

class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        super(Model, self).__init__()

        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)#.type(torch.FloatTensor)
        print("Number of features: ", len(features))
        return features


    def build_input_layer(self):
        return RelAttLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RelAttLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RelAttLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(F.softmax, dim=1))

    def forward(self, g):
        logging.debug("In model forward")
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            logging.info("New layer")
            layer(g)
        return g.ndata.pop('h')

###############################################################################
# Handle dataset
# ~~~~~~~~~~~~~~~~
# This tutorial uses Institute for Applied Informatics and Formal Description Methods (AIFB) dataset from R-GCN paper.

# load graph data
from dgl.contrib.data import load_data
import numpy as np
data = load_data(dataset='aifb')
num_nodes = data.num_nodes
num_rels = data.num_rels
num_classes = data.num_classes
labels = data.labels
train_idx = data.train_idx
# split training and validation set
val_idx = train_idx[:len(train_idx) // 5]
train_idx = train_idx[len(train_idx) // 5:]

# edge type and normalization factor
edge_type = torch.from_numpy(data.edge_type)
edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)

num_edges = len(edge_type)

labels = torch.from_numpy(labels).view(-1)

h_n = torch.arange(data.num_nodes * data.num_nodes).type(torch.FloatTensor)
h_n = h_n.view(data.num_nodes, data.num_nodes)

#h_e = torch.arange(num_edges* data.num_nodes).type(torch.FloatTensor)
#h_e = h_e.view(num_edges, data.num_nodes)

h_e = torch.arange(num_rels* data.num_nodes).type(torch.FloatTensor)
h_e = h_e.view(num_rels, data.num_nodes)


    


logging.debug("h_nodes " + str(h_n.shape))
logging.debug("h_rel " + str(h_e.shape))

###############################################################################
# Create graph and model
# ~~~~~~~~~~~~~~~~~~~~~~~

# configurations
n_hidden = 16 # number of hidden units
n_bases = -1 # use number of relations as number of bases
n_hidden_layers = 0 # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 25 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient

# create graph
g = DGLGraph((data.edge_src, data.edge_dst))
#g.ndata['feat'] = h_n
g.edata.update({'rel_type': edge_type, 'norm': edge_norm})

set_types = list(set(data.edge_type))
for i in range(len(set_types)):
    curr_type = set_types[i]

    logging.debug("he " + str(h_e[i].shape))
    g.edges[curr_type].data['feat'] = torch.unsqueeze(h_e[i],0)


# create model
model = Model(len(g),
              n_hidden,
              num_classes,
              num_rels,
              num_bases=n_bases,
              num_hidden_layers=n_hidden_layers)

###############################################################################
# Training loop
# ~~~~~~~~~~~~~~~~

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

print("start training...")
model.train()
for epoch in range(n_epochs):
    optimizer.zero_grad()

    logits = model.forward(g)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])
    loss.backward()

    optimizer.step()

    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx])
    train_acc = train_acc.item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
    val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx])
    val_acc = val_acc.item() / len(val_idx)
    print("Epoch {:05d} | ".format(epoch) +
          "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
              train_acc, loss.item()) +
          "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
              val_acc, val_loss.item()))

###############################################################################
# .. _link-prediction:
#
# The second task, link prediction
# --------------------------------
# So far, you have seen how to use DGL to implement entity classification with an 
# R-GCN model. In the knowledge base setting, representation generated by
# R-GCN can be used to uncover potential relationships between nodes. In the 
# R-GCN paper, the authors feed the entity representations generated by R-GCN
# into the `DistMult <https://arxiv.org/pdf/1412.6575.pdf>`_ prediction model
# to predict possible relationships.
#
# The implementation is similar to that presented here, but with an extra DistMult layer
# stacked on top of the R-GCN layers. You can find the complete
# implementation of link prediction with R-GCN in our `Github Python code example
#  <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/link_predict.py>`_.

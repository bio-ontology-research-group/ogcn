"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction
Difference compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR. User could specify "--edge-sampler=neighbor" to switch
  to neighbor-based edge sampling.
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# from dgl.data.knowledge_graph import load_data
from dgl.contrib.data import load_data

from relGraphConv import RelGraphConv

import logging

logging.basicConfig(level=logging.DEBUG)

from baseRGCN import BaseRGCN

import utils


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, num_rels, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.n_embedding = torch.nn.Embedding(num_nodes, h_dim)
        self.e_embedding = torch.nn.Embedding(num_rels, h_dim)
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.h_dim = h_dim

    def forward(self, g, hn, r, he, norm):

        logging.debug("Embedding HN: " + str(hn.shape) + "  " + str(self.num_nodes) + "  " + str(self.h_dim))
        logging.debug("Embedding HE: " + str(he.shape) + "  " + str(self.num_rels))
        logging.debug("Embedding HN: " + str(self.n_embedding(hn.squeeze()).shape) + "  " + str(self.num_nodes))
        logging.debug("Squeeze HE: " + str(he.squeeze().shape))
        logging.debug("Embedding HE: " + str(self.e_embedding(he.squeeze()).shape) + "  " + str(self.num_rels))
        return self.n_embedding(hn.squeeze()), self.e_embedding(he.squeeze())


# class EmbeddingLayer(nn.Module):
#     def __init__(self, num_nodes, num_rels, h_dim):
#         super(EmbeddingLayer, self).__init__()
#         self.n_embedding = torch.nn.Embedding(num_nodes, h_dim)
#         self.e_embedding = torch.nn.Embedding(num_rels, h_dim)

#     def forward(self, g, hn, r, he, norm):

#         logging.debug("Embedding HN: " + str(hn.shape))
#         logging.debug("Embedding HE: " + str(he.shape))
#         return self.n_embedding(hn.squeeze()), self.e_embedding(he.squeeze())

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.num_rels,self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout, low_mem = True)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels,num_bases=-1,
                 num_hidden_layers=1, dropout=0, attn_drop = 0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, he, norm):
        
        logging.debug("Link predict HN: " + str(h.shape))
        logging.debug("Link predict HE: " + str(he.shape))

        return self.rgcn.forward(g, h, r, he, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        #logging.debug("score: " + str(score))
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

def main(args):
    # load graph data
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    train_data = data.train
    num_edges = len(train_data)
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    logging.debug("Num rels: " + str(num_rels))

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        attn_drop=args.attn_drop,
                        use_cuda=use_cuda,
                        reg_param=args.regularization)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build test graph
    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_edge_feat = torch.arange(0, num_rels*2, dtype=torch.long).view(-1, 1)
      
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_statewn.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    while True:
        torch.set_grad_enabled(True)
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)
        print("Done edge sampling")

        logging.debug("Node id: " + str(node_id.shape))
        logging.debug("Edge type: " + str(edge_type.shape))

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_feat = torch.arange(num_rels*2).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)

        logging.debug("Node id: " + str(node_id.shape))
        logging.debug("Edge type: " + str(edge_type.shape))



        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        logging.debug("Edge norm: " + str(edge_norm.shape))


        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm, edge_feat = edge_type.cuda(), edge_norm.cuda(), edge_feat.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)

        t0 = time.time()
        logging.debug("Node id: "   + str(type(node_id)) + "   " + str(node_id.shape))
        logging.debug("Edge type: " + str(type(edge_feat)) + "   " + str(edge_feat.shape))

        embed, e_embed = model(g, node_id, edge_type, edge_feat, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            torch.set_grad_enabled(False)
            model.eval()
            print("start eval")
            embed, e_embed = model(test_graph, test_node_id, test_rel, test_edge_feat,test_norm)
            mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                                 valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                                 eval_p=args.eval_protocol)
            # save best model
            if best_mrr < mrr:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            
            if epoch >= args.n_epochs:
                break
            
            if use_cuda:
                model.cuda()

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu() # test on CPU
    torch.set_grad_enabled(False)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed, e_embed = model(test_graph, test_node_id, test_rel, test_edge_feat,test_norm)
    utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
                   test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--attn-drop", type=float, default=0.2,
            help="attention dropout probability")
    parser.add_argument("--n-hidden", type=int, default=504,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='wn18', #FB15k-237
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="filtered",
            help="type of evaluation protocol: 'raw' or 'filtered' mrr")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
            help="type of edge sampler: 'uniform' or 'neighbor'")

    args = parser.parse_args()
    print(args)
    main(args)
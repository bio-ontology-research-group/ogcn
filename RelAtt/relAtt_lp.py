import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import graph
import dgl.function as fn
from functools import partial

import logging
logging.basicConfig(level=logging.DEBUG)

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

        if self.is_input_layer:
            h_src = edges.src['feat']
            z_src = self.shared_weight(h_src)
            h_dst = edges.dst['feat']
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
            logging.debug("locator ")
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
        # h2o = self.build_output_layer()
        # self.layers.append(h2o)

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

    # def build_output_layer(self):
    #     return RelAttLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
    #                      activation=partial(F.softmax, dim=1))


    def forward(self, g):
        logging.debug("In model forward")
      #  if self.features is not None:
      #      g.ndata['id'] = self.features
        for layer in self.layers:
            logging.info("New layer")
            layer(g)
        return g.ndata.pop('h')

###############################################################################
# Handle dataset
# ~~~~~~~~~~~~~~~~
# This tutorial uses Institute for Applied Informatics and Formal Description Methods (AIFB) dataset from R-GCN paper.
class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())

from model import BaseRGCN
import argparse
from dgl.contrib.data import load_data
import utils
import time


# class RGCN(BaseRGCN):

#     def build_input_layer(self):
#         return RelAttLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases,
#                          activation=F.relu, is_input_layer=True)

#     def build_hidden_layer(self):
#         return RelAttLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                          activation=F.relu)

#     def build_output_layer(self):
#         return RelAttLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
#                          activation=partial(F.softmax, dim=1))

    # def build_input_layer(self):
    #     return RelAttLayer(self.num_nodes, self.h_dim)

    # def build_hidden_layer(self, idx):
    #     #act = F.relu if idx < self.num_hidden_layers - 1 else None
    #     return RelAttLayer(self.h_dim, self.h_dim, self.num_rels, "bdd",
    #             self.num_bases, activation=act, self_loop=True,
    #             dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = Model(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers) #, dropout, use_cuda)
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

    def forward(self, g): #, h, r, norm):
        return self.rgcn.forward(g)#, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
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
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        logging.info("Using GPU")
        torch.cuda.set_device(args.gpu)

    # create model
    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
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
    test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)
        print("Done edge sampling")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)


        if use_cuda:

            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()
            g = g.to(args.gpu)

        g.ndata['id'] = node_id

        t0 = time.time()
        embed = model(g)#, node_id, edge_type, edge_norm)
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
            model.eval()
            print("start eval")
            embed = model(test_graph, test_node_id, test_rel, test_norm)
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
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(test_graph, test_node_id, test_rel, test_norm)
    utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data), valid_data,
                   test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size, eval_p=args.eval_protocol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=6000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='FB15k',
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
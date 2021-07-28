from functools import partial
import click as ck
import numpy as np
import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import dgl
from ppi_gcn_rel import load_data, train, load_graph_data

@ck.command()
@ck.option(
    '--train-inter-file', '-trif', default='data/4932.train_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--test-inter-file', '-tsif', default='data/4932.test_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--data-file', '-df', default='data/swissprot.pkl',
    help='Data file with protein sequences')
@ck.option(
    '--deepgo-model', '-dm', default='data/deepgoplus.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--model-file', '-mf', default='data/9606.model.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')

def main(train_inter_file, test_inter_file, data_file, deepgo_model, model_file, epochs, load, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    
    #global g, annots
    device = 'cuda'
   
    g, annots, prot_idx = load_graph_data(data_file)
    
    num_nodes = g.number_of_nodes()
    print(f"Num nodes: {g.number_of_nodes()}")
    
    annots = th.FloatTensor(annots).to(device)
    num_rels = len(g.canonical_etypes)

    g = dgl.to_homogeneous(g)

    num_bases = 20
    feat_dim = 2
    loss_func = nn.BCELoss()

    
    tuning(g, annots, epochs, data_file, train_inter_file, test_inter_file)

def train_tune(config, g=None, annots=None, epochs=None, data_file=None, train_inter_file=None, test_inter_file=None, checkpoint_dir = None):
    batch_size = config["batch_size"]
    train(g, annots, prot_idx, feat_dim, num_rels, num_bases, num_nodes, loss_func, device, batch_size, epochs, data_file, train_inter_file, test_inter_file)


def tuning(g, annots, epochs, data_file, train_inter_file, test_inter_file, num_samples=1, max_num_epochs=1, gpus_per_trial=1):
    
    load_data(train_inter_file, test_inter_file)
    
    config = {
        "n_layers": tune.choice([1, 2, 3]),
        "dropout": tune.choice([x/10 for x in range(1,9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32])
    }
    scheduler = ASHAScheduler(
        metric="auc",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "auc"])
    result = tune.run(
        tune.with_parameters(train_tune, 
                                g=g, 
                                annots=annots, 
                                epochs=epochs, 
                                data_file=data_file, 
                                train_inter_file = train_inter_file, 
                                test_inter_file = test_inter_file),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("auc", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if th.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = th.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))



if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main()

from functools import partial
import click as ck
import numpy as np
import os
import torch as th
import torch.nn as nn
import dgl
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


from ppi_gcn_rel import load_data, train, test, load_graph_data, PPIModel, GraphDataset, get_batches


import os
curr_path = os.path.dirname(os.path.abspath(__file__))


def main(num_samples, max_num_epochs, gpus_per_trial):
    
    #global g, annots
    file_params = {
        "data_file": curr_path + '/data/swissprot.pkl',
        "train_inter_file": curr_path + '/data/4932.train_interactions.pkl',
        "test_inter_file": curr_path + '/data/4932.test_interactions.pkl',
        "graph_file": curr_path + '/data/go_subclass.bin',
        "nodes_file": curr_path + '/data/go_subclass.pkl'
    }

    feat_dim = 2
    tuning(file_params, num_samples, max_num_epochs, gpus_per_trial, feat_dim)

def train_tune(config, file_params=None, checkpoint_dir = None):
    batch_size = config["batch_size"]
    n_hid = config["n_hid"]
    dropout = config["dropout"]
    lr = config["lr"]
    num_bases = config["num_bases"]


    epochs = 32
    train(n_hid, dropout, lr, num_bases, batch_size, epochs, file_params, tuning = True)


def tuning(file_params, num_samples, max_num_epochs, gpus_per_trial, feat_dim):

    data_file = file_params["data_file"]
    train_inter_file = file_params["train_inter_file"]
    test_inter_file = file_params["test_inter_file"]


    g, _, _ = load_graph_data(file_params)
    num_rels = len(g.canonical_etypes)
    g = dgl.to_homogeneous(g)
    num_nodes = g.number_of_nodes()

    
    #load_data(file_params)
    
    config = {
        "n_hid": tune.choice([1, 2, 3]),
        "dropout": tune.choice([x/10 for x in range(1,9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32]),
        "num_bases": tune.choice([1])
    }
    scheduler = ASHAScheduler(
        metric="auc",
        mode="max",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "auc"])
    result = tune.run(
        tune.with_parameters(train_tune,
                                file_params=file_params),
        resources_per_trial={"cpu": gpus_per_trial, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("auc", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation auc: {}".format(
        best_trial.last_result["auc"]))

    best_trained_model = PPIModel(feat_dim, num_rels, best_trial.config["num_bases"], num_nodes, best_trial.config["n_hid"], best_trial.config["dropout"])
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

    test_loss, test_auc = test(best_trial.config["n_hid"], best_trial.config["dropout"], best_trial.config["num_bases"], best_trial.config["batch_size"], file_params, model = best_trained_model)
    print("Best trial test set loss: {}".format(test_loss))
    print("Best trial test set auc: {}".format(test_auc))
   


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    num_samples = 100
    max_num_epochs = 10
    gpus_per_trial = 1

    main(num_samples, max_num_epochs, gpus_per_trial)

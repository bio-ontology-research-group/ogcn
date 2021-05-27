#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import Ontology, is_exp_code, is_cafa_target, FUNC_DICT

logging.basicConfig(level=logging.INFO)

ORG_ID = '9606'

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot.pkl',
    help='Result file with a list of proteins, sequences and annotations')
@ck.option(
    '--inter-file', '-if', default=f'data/{ORG_ID}.protein.links.detailed.v11.0.txt.gz',
    help='Data file with protein sequences')
@ck.option(
    '--neg-file', '-nif', default=f'data/{ORG_ID}.neg_interactions.txt.gz',
    help='Data file with protein sequences')
@ck.option(
    '--train_out-file', '-trof', default=f'data/{ORG_ID}.train_interactions.pkl',
    help='Data file with protein sequences')
@ck.option(
    '--test-out-file', '-tsof', default=f'data/{ORG_ID}.test_interactions.pkl',
    help='Data file with protein sequences')
def main(data_file, inter_file, neg_file, train_out_file, test_out_file):
    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '9606']
    uni_pros = set()
    st_pros = set()
    st2uni = {}
    for i, row in enumerate(df.itertuples()):
        uni_ids = [x.strip() for x in row.accessions.split(';')][:1]
        uni_pros |= set(uni_ids)
        st_pros |= set(row.string_ids)
        for st_id in row.string_ids:
            st2uni[st_id] = uni_ids[0]
    proteins = np.array(list(st2uni.values()))
    n = len(proteins)
    pindex = np.arange(n)
    np.random.shuffle(pindex)
    train_n = int(n * 0.95)
    train_prots = set(proteins[pindex[:train_n]])
    
    pos_interactions_train = set()
    pos_interactions_test = set()
    with gzip.open(inter_file, 'rt') as f:
        next(f)
        for line in f:
            it = line.strip().split()
            p1 = it[0]
            p2 = it[1]
            score = float(it[6])
            # Ignore zero experimental score
            if score == 0:
                continue
            # Ignore proteins without sequence info
            if p1 not in st2uni or p2 not in st2uni:
                continue
            p1, p2 = st2uni[p1], st2uni[p2]
            if p1 == p2:
                continue
            if p1 in train_prots and p2 in train_prots:
                if (p2, p1) not in pos_interactions_train and (p1, p2) not in pos_interactions_train:
                    pos_interactions_train.add((p1, p2))
            else:
                if (p2, p1) not in pos_interactions_test and (p1, p2) not in pos_interactions_test:
                    pos_interactions_test.add((p1, p2))
    
    neg_interactions_train = set()
    neg_interactions_test = set()
    with gzip.open(neg_file, 'rt') as f:
        for line in f:
            it = line.strip().split()
            p1 = it[0][10:].strip()
            p2 = it[1][10:].strip()
            # Ignore proteins without sequence info
            if p1 not in uni_pros or p2 not in uni_pros:
                continue
            if p1 == p2:
                continue
            if p1 in train_prots and p2 in train_prots:
                if (p2, p1) not in neg_interactions_train and (p1, p2) not in neg_interactions_train:
                    neg_interactions_train.add((p1, p2))
            else:
                if (p2, p1) not in neg_interactions_test and (p1, p2) not in neg_interactions_test:
                    neg_interactions_test.add((p1, p2))
        
    # remove positive interactions from negatives
    for p1, p2 in pos_interactions_train:
        neg_interactions_train.discard((p1, p2))
        neg_interactions_train.discard((p2, p1))
    for p1, p2 in pos_interactions_test:
        neg_interactions_test.discard((p1, p2))
        neg_interactions_test.discard((p2, p1))

    print(len(pos_interactions_train), len(neg_interactions_train))
    print(len(pos_interactions_test), len(neg_interactions_test))
    train_interactions = list(pos_interactions_train) + list(neg_interactions_train)
    train_labels = [1] * len(pos_interactions_train) + [0] * len(neg_interactions_train)
    df = pd.DataFrame({'interactions': train_interactions, 'labels': train_labels})
    df.to_pickle(train_out_file)
    print(df)
    test_interactions = list(pos_interactions_test) + list(neg_interactions_test)
    test_labels = [1] * len(pos_interactions_test) + [0] * len(neg_interactions_test)
    df = pd.DataFrame({'interactions': test_interactions, 'labels': test_labels})
    df.to_pickle(test_out_file)
    
if __name__ == '__main__':
    main()

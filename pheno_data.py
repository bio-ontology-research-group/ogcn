#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip
import logging
from utils import Ontology, is_exp_code, is_cafa_target, FUNC_DICT
from collections import Counter

logging.basicConfig(level=logging.INFO)

ORG_ID = '9606'

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_human.pkl',
    help='Result file with a list of proteins, sequences and annotations')
@ck.option(
    '--inter-file', '-if', default=f'data/{ORG_ID}.protein.links.detailed.v11.0.txt.gz',
    help='Data file with protein sequences')
@ck.option('--pheno-file', default=f'data/genes_to_phenotype.txt', help='')
@ck.option(
    '--out-file', '-of', default=f'data/data_human.pkl',
    help='Data file with protein sequences')
def main(data_file, inter_file, pheno_file, out_file):
    hpo = Ontology('data/hp.obo')
    df = pd.read_pickle(data_file)
    uni_pros = set()
    st_pros = set()
    st2uni = {}
    for i, row in enumerate(df.itertuples()):
        uni_ids = [x.strip() for x in row.accessions.split(';')][:1]
        prot_id = row.proteins
        uni_pros |= set(uni_ids)
        st_pros |= set(row.string_ids)
        for st_id in row.string_ids:
            st2uni[st_id] = prot_id
    proteins = set(df['proteins'].values)
    
    interactions = {}
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
            if p1 not in interactions:
                interactions[p1] = []
            if p2 not in interactions:
                interactions[p2] = []
            interactions[p1].append(p2)
            interactions[p2].append(p1)
    inters = []
    for i, row in df.iterrows():
        prot_id = row.proteins
        if prot_id in interactions:
            inters.append(interactions[prot_id])
        else:
            inters.append([])

    phenos = {}
    with open(pheno_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            it = line.strip().split('\t')
            gene_id = it[0]
            hp_id = it[2]
            if gene_id not in phenos:
                phenos[gene_id] = []
            phenos[gene_id].append(hp_id)

    phenotypes = []
    prop_phenotypes = []
    go_cnt = Counter()
    hp_cnt = Counter()

    index = []
    for i, row in df.iterrows():
        gene_id = row.genes
        if gene_id in phenos:
            phenotypes.append(phenos[gene_id])
            prop_phenos = hpo.get_prop_terms(phenos[gene_id])
            prop_phenotypes.append(prop_phenos)
            hp_cnt.update(prop_phenos)
            go_cnt.update(row.prop_annotations)
        else:
            phenotypes.append([])
            prop_phenotypes.append([])
    df['interactions'] = inters
    df['phenotypes'] = phenotypes
    df['prop_phenotypes'] = prop_phenotypes

    df.to_pickle(out_file)

    del hp_cnt['HP:0000001']
    for go_id in FUNC_DICT.values():
        del go_cnt[go_id]

    hp_terms = [hp_id for hp_id, cnt in hp_cnt.items() if cnt >= 10]
    go_terms = [go_id for go_id, cnt in go_cnt.items()]
    go_df = pd.DataFrame({'terms': go_terms})
    go_df.to_csv('data/go_terms.csv')
    hp_df = pd.DataFrame({'terms': hp_terms})
    hp_df.to_csv('data/hp_terms.csv')

    print(len(hp_terms), len(go_terms))
    
    
if __name__ == '__main__':
    main()

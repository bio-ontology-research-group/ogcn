#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging

logging.basicConfig(level=logging.INFO)

ORG_ID = '9606'

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_2021_03.pkl',
    help='Pandas dataframe with protein sequences')
@ck.option(
    '--out-file', '-o', default=f'data/{ORG_ID}.annotations.tsv',
    help='Fasta file')
@ck.option(
    '--terms-file', '-o', default=f'data/{ORG_ID}.terms.tsv',
    help='Fasta file')
def main(data_file, out_file, terms_file):
    # Load interpro data
    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '9606']
    print(len(df))
    terms = set()
    with open(out_file, 'w') as f:
        for row in df.itertuples():
            f.write(row.accessions.split(';')[0].strip())
            for go_id in row.prop_annotations:
                f.write('\t' + go_id)
                terms.add(go_id)
            f.write('\n')

    terms = list(terms)
    with open(terms_file, 'w') as f:
        for t_id in terms:
            f.write(t_id + '\n')

if __name__ == '__main__':
    main()

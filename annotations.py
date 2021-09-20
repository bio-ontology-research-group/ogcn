#!/usr/bin/env python

import click as ck
import numpy as np
import pandas as pd
import gzip

from collections import Counter
from aminoacids import MAXLEN, to_ngrams
import logging

logging.basicConfig(level=logging.INFO)

@ck.command()
@ck.option(
    '--data-file', '-df', default='data/swissprot_2021_03.pkl',
    help='Pandas dataframe with protein sequences')
@ck.option(
    '--out-file', '-o', default='data/4932.annotations.tsv',
    help='Fasta file')
def main(data_file, out_file):
    # Load interpro data
    df = pd.read_pickle(data_file)
    df = df[df['orgs'] == '559292']
    print(len(df)) 
    with open(out_file, 'w') as f:
        for row in df.itertuples():
            f.write(row.accessions.split(';')[0].strip())
            for go_id in row.prop_annotations:
                f.write('\t' + go_id)
            f.write('\n')
    

if __name__ == '__main__':
    main()

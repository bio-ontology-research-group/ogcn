from collections import deque, Counter
import warnings
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
import math
import utils as u
import dgl

class Ontology(object):

    def __init__(self, filename='data/go.obo', rels=[], with_disjoint=False):
        self.rels = rels
        self.with_disjoint = with_disjoint
        self.ont = self.load(filename, self.rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, rels=[]):
        ont = dict()

        with open(filename, 'r') as f:
            chunk = []
            for line in f:
                line = line.strip()
                if not line: #empty line
                    obj = self.processChunk(chunk, rels)
                    if obj != None:  
                        ont[obj['id']] = obj
                    chunk = []
                    continue
                
                else:
                    chunk.append(line)

        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        return ont        
        

    def toDGLGraph(self):
        # Consider that there is only one type of nodes 
        ######
        edges = dict()
        edges[('node', 'is_a', 'node')] = list()
        for rel in self.rels:
            edges[('node', rel, 'node')] = list()
        if self.with_disjoint:
            edges[('node', 'disjoint_from', 'node')] = list()


        nodes = list(self.ont.keys())
        node_idx = {v: k for k, v in enumerate(nodes)}

        for n_id in nodes:
            src = node_idx[n_id]

            #is_a relation
            for dst_id in self.ont[n_id]['is_a']:
                dst = node_idx[dst_id]
                edges[('node', 'is_a', 'node')].append([src, dst])

            #other relations
            for rel in self.rels:
                if rel in self.ont[n_id]:
                    for dst_id in self.ont[n_id][rel]:
                        dst = node_idx[dst_id]
                        edges[('node', rel, 'node')].append([src, dst])

            if self.with_disjoint:
                #disjoint_from relation
                for dst_id in self.ont[n_id]['disjoint_from']:
                    dst = node_idx[dst_id]
                    edges[('node', 'disjoint_from', 'node')].append([src, dst])


        print(('node', 'is_a', 'node'), len(edges[('node', 'is_a', 'node')]))
        for rel in self.rels:
            print(('node', rel, 'node'), len(edges[('node', rel, 'node')]))
        if self.with_disjoint:
            print(('node', 'disjoint_from', 'node'), len(edges[('node', 'disjoint_from', 'node')]))


        return dgl.heterograph(edges)

    def processChunk(self, chunk, rels=[]):
        if chunk[0] != '[Term]':
            return None
        
        obj = dict()

        for rel in rels:
            obj[rel] = list()
        
        obj['is_a'] = list()
        obj['alt_ids'] = list()
        obj['is_obsolete'] = False
        if self.with_disjoint:
            obj['disjoint_from'] = list()
        
        for line in chunk[1:]:
            key, val = tuple(line.split(": ")[:2])
            if key == 'id':
                obj['id'] = val
            elif key == 'alt_id':
                obj['alt_ids'].append(val)
            elif key == 'is_a':
                obj['is_a'].append(val.split(' ! ')[0])
            elif key == 'relationship':
                rel, val = val.split()[:2]
                if rel in rels:
                    obj[rel].append(val)
            elif key == 'is_obsolete' and val == 'true':
                obj['is_obsolete'] = True
            elif self.with_disjoint and key == 'disjoint_from':
                obj['disjoint_from'].append(val.split(' ! ')[0])
        return obj

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()

        for term_id in terms:
            prop_terms |= self.get_anchestors(term_id)
        return prop_terms


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

def read_fasta(filename):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    info.append(inf)
                    seq = ''
                inf = line[1:]
            else:
                seq += line
        seqs.append(seq)
        info.append(inf)
    return info, seqs


class DataGenerator(object):

    def __init__(self, batch_size, is_sparse=False):
        self.batch_size = batch_size
        self.is_sparse = is_sparse

    def fit(self, inputs, targets=None):
        self.start = 0
        self.inputs = inputs
        self.targets = targets
        if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
            self.size = self.inputs[0].shape[0]
        else:
            self.size = self.inputs.shape[0]
        self.has_targets = targets is not None

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    def next(self):
        if self.start < self.size:
            batch_index = np.arange(
                self.start, min(self.size, self.start + self.batch_size))
            if isinstance(self.inputs, tuple) or isinstance(self.inputs, list):
                res_inputs = []
                for inp in self.inputs:
                    if self.is_sparse:
                        res_inputs.append(
                            inp[batch_index, :].toarray())
                    else:
                        res_inputs.append(inp[batch_index, :])
            else:
                if self.is_sparse:
                    res_inputs = self.inputs[batch_index, :].toarray()
                else:
                    res_inputs = self.inputs[batch_index, :]
            self.start += self.batch_size
            if self.has_targets:
                if self.is_sparse:
                    labels = self.targets[batch_index, :].toarray()
                else:
                    labels = self.targets[batch_index, :]
                return (res_inputs, labels)
            return res_inputs
        else:
            self.reset()
            return self.next()


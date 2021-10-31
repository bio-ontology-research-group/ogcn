# Sarah M. Alghamdi
#------------------------------------
# this code is used to generate input for evaluting gene-disease similarities generated from DL2Vec tool, Resnik's sililarty and OWL2Vec* 
#------------------------------------
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
#------------------------------------
import json
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
#------------------------------------
# generate similarity  matrex and input for DL2Vec and OWL2Vec*

'''
init = sys.argv[1]
embedding_model = sys.argv[2]
word2vec_model=gensim.models.Word2Vec.load(embedding_model)
word_vocabs = word2vec_model.wv.vocab

HGs={}
HDs={}

for key in word_vocabs:
    if(("MGI" in key)):
        HGs[key] = word2vec_model[key].tolist()

    #elif("OMIM" in key and "http://url" in key):
    elif("OMIM" in key and "http" not in key):
        new_key = key.replace("http://url/","")
        HDs[new_key] = word2vec_model[key].tolist()


with open(init+'HG.json', 'w') as fp:
        json.dump(HGs, fp)
with open(init+'HD.json', 'w') as fp:
        json.dump(HDs, fp)
# compute the gene disease similarity matrices
HDs_vectors=[]
HDs_keys = list(HDs.keys())
print(len(HDs_keys))
for key in HDs_keys:
    HDs_vectors.append(HDs[key])        

with open(init+'_d_keys.json', 'w') as fp:
    json.dump(HDs_keys, fp)

HGs_vectors=[]
HGs_keys = list(HGs.keys())

for key in HGs_keys:
    HGs_vectors.append(HGs[key])
print(len(HGs_keys))
with open(init+'_g_keys.json', 'w') as fp:
    json.dump(HGs_keys, fp)

print(len(HGs_vectors), len(HDs_vectors))
OGs_HDs= cosine_similarity(np.array(HGs_vectors),np.array(HDs_vectors))
np.save(init+'_dl_cosine_sim.npy',OGs_HDs)





'''

# generate similarity matrex and input for resnik's similarity

'''
sim_scores = np.loadtxt(sys.argv[1], dtype = 'float32')
diseases = np.genfromtxt(sys.argv[2], dtype = 'str')
genes = np.genfromtxt(sys.argv[3], dtype = 'str')

sim_mat = sim_scores.reshape((len(genes),len(diseases)))

np.save(sys.argv[4]+"_resnik_sim",sim_mat)
with open(sys.argv[4]+'_resnik_g_keys.json', 'w') as fp:
    json.dump(genes.tolist(), fp)
with open(sys.argv[4]+'_resnik_d_keys.json', 'w') as fp:
    json.dump(diseases.tolist(), fp)

'''

# Implementation of the paper [Knowledge Graph Embedding using Graph Convolutional Networks with Relation-Aware Attention](https://arxiv.org/pdf/2102.07200.pdf)



## Results

### Link prediction

#### __FB15k-237__


* attention dropout: 0.1
* number of hidden units: 500
* regularization: bdd
* learning rate: 3e-10

| Model             | MRR       | Hits@1    | Hits@3    | Hits@10   |
|-------------------|-------    |--------   |--------   |---------  |
| RGCN (paper)      | 0.226     | 0.126     | 0.252     | 0.421     |
| RelAtt (paper)    | 0.234     | 0.139     | __0.258__ | __0.428__ |
| RGCN (this)       | 0.233     | 0.145     | 0.255     | 0.416     |
| RelAtt (this)     | __0.236__ | __0.146__ | 0.256     | 0.425     |


#### __WN18__

* attention dropout: 0.1
* number of hidden units: 504
* regularization: bdd
* learning rate: 3e-10



| Model             | MRR   | Hits@1 | Hits@3 | Hits@10 |
|-------------------|-------|--------|--------|---------|
| RGCN (paper)      | 0.750 | 0.612  | 0.882  | 0.939   |
| RelAtt (paper)    | 0.767 | 0.639  | 0.889  | 0.940   |
| RGCN (this)       | 0.783 | 0.664  | 0.897  | 0.939   |
| RelAtt (this)     | 0.778 | 0.657  | 0.894  | 0.939   |


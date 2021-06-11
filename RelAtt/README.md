# Implementation of the paper [Knowledge Graph Embedding using Graph Convolutional Networks with Relation-Aware Attention](https://arxiv.org/pdf/2102.07200.pdf)



## Results

### Link prediction

| Model             | MRR   | Hits@1 | Hits@3 | Hits@10 |
|-------------------|-------|--------|--------|---------|
| RGCN (paper)      | 0.226 | 0.126  | 0.252  | 0.421   |
| RelAtt (paper)    | 0.234 | 0.139  | 0.258  | 0.428   |
| RGCN (this)       | 0.228 | 0.141  | 0.251  | 0.408   |
| RelAtt (this)     | 0.236 | 0.146  | 0.256  | 0.425   |
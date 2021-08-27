
import os
import pickle as pkl
import logging

import dgl

logging.basicConfig(level=logging.DEBUG)

#JPype imports
import jpype
import jpype.imports
jars_dir = "../gateway/build/distributions/gateway/lib/"
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jars,
        convertStrings=False)

from org.ogcn.parse import Parser
from org.ogcn.parse import Types

def main():
    logging.info(f"Top: {Types.goClassToStr(Types.Top())}")
    logging.info(f"Top: {Types.goClassToStr(Types.Bottom())}")

    parser = Parser("../data/go-plus.owl")

    edges = parser.parse()

    nodes = list({str(e.src()) for e in edges}.union({str(e.dst()) for e in edges}))


    node_idx = {v: k for k, v in enumerate(nodes)}
    logging.info(f"Top: {node_idx['owl#Thing']}")
    logging.info(f"Top: {node_idx['owl#Nothing']}")

    logging.info(f"Number of edges: {len(edges)}")

    logging.debug(f"First edge: {edges[0]}")
    
    graph = {}
    for edge in edges:
        go_class_1 = edge.src()
        rel = str(edge.rel())
        go_class_2 = edge.dst()
        
        key = ("node", rel, "node")
        if not key in graph:
            graph[key] = set()

        node1 = node_idx[go_class_1]
        node2 = node_idx[go_class_2]

        graph[key].add((node1, node2))
    

    graph = {k: list(v) for k, v in graph.items()}

    rels = {k: len(v) for k, v in graph.items()}

    logging.info(f"Number of nodes: {len(nodes)}")
    logging.info(f"Number of rels: {len(rels)}")
    logging.info(f"Edges in the graph:\n{rels}")

    graph = dgl.heterograph(graph)

    dgl.save_graphs("../data/go_cat3.bin", graph)


    logging.debug(f"Type of node_idx: {type(node_idx)}")
   
    with open("../data/nodes_cat3.pkl", "wb") as pkl_file:
        pkl.dump(node_idx, pkl_file)



if __name__ == '__main__':
    main()


import os
import pickle as pkl
import logging


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

def main():

    parser = Parser("/home/zhapacfp/Github/ogcn/data/go.owl")

    edges = parser.parse()

    logging.info(f"Number of edges: {len(edges)}")

    logging.debug(f"First edge: {edges[0]}")
    graph = {}
    for go_class_1, rel, go_class_2 in edges:
        key = ("node", rel, "node")
        if not key in graph:
            graph[key] = list()

        node1 = node_idx[go_class_1]
        node2 = node_idx[go_class_2]

        graph[key].append([node1, node2])
    
    graph = {k: v for k, v in graph.items() if len(v) > 100}
   
    rels = {k: len(v) for k, v in graph.items()}

    logging.info(f"Number of nodes: {len(go_classes)}")
    logging.info(f"Edges in the graph:\n{rels}")

    graph = dgl.heterograph(graph)

    dgl.save_graphs("../data/go_cat3.bin", graph)


    logging.debug(f"Type of node_idx: {type(node_idx)}")
    node_idx = {prettyFormat(v): k for k, v in enumerate(go_classes)}
    
    with open("../data/nodes_cat3.pkl", "wb") as pkl_file:
        pkl.dump(node_idx, pkl_file)



if __name__ == '__main__':
    main()

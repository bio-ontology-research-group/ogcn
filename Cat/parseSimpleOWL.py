
import os
import pickle as pkl
import logging
import click as ck
import dgl

#logging.basicConfig(filename='../data/logtmp.txt', filemode = 'w', level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

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

from org.ogcn.parse import SimpleParser
from org.ogcn.parse import Types



@ck.command()
@ck.option(
    '--in-file', '-i',
    help='Ontology input file in .owl format')
@ck.option(
    '--out-file', '-o',
    help='Name of output file (with no extension)')



def main(in_file, out_file):

    parser = SimpleParser(f"{in_file}", True, True)

    edges = parser.parse()

    nodes = list({str(e.src()) for e in edges}.union({str(e.dst()) for e in edges}))


    node_idx = {v: k for k, v in enumerate(nodes)}

    logging.info(f"Number of edges: {len(edges)}")
    
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

    dgl.save_graphs(f"{out_file}.bin", graph)


    logging.debug(f"Type of node_idx: {type(node_idx)}")
   
    with open(f"{out_file}.pkl", "wb") as pkl_file:
        pkl.dump(node_idx, pkl_file)



if __name__ == '__main__':
    main()

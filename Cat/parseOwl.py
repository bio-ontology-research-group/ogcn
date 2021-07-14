

import os
import pickle as pkl
import logging


logging.basicConfig(level=logging.INFO)


#JPype imports
import jpype
import jpype.imports
jars_dir = "../jars/"
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jars,
        convertStrings=False)

from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model.parameters import Imports
from java.io import File

#DGL imports
import dgl

def main():
    ont_manager = OWLManager.createOWLOntologyManager()
    data_factory = ont_manager.getOWLDataFactory()

    global ontology 
    ontology = ont_manager.loadOntologyFromOntologyDocument(File("../data/go.owl"))
    
    axioms = ontology.getAxioms()

    imports = Imports.fromBoolean(False)
    go_classes = ontology.getClassesInSignature(imports)
    node_idx = {v: k for k, v in enumerate(go_classes)}

    edges = []
    for go_class in go_classes:
        new_edges = processAxioms(go_class)
        edges += new_edges

    logging.debug(f"Number of edges: {len(edges)}")
    logging.debug(f"First edge: {edges[0]}")
    graph = {}
    for go_class_1, rel, go_class_2 in edges:
        key = ("node", rel, "node")
        if not key in graph:
            graph[key] = list()

        node1 = node_idx[go_class_1]
        logging.debug(f"goClass2: {go_class_2}")
        node2 = node_idx[go_class_2]

        graph[key].append([node1, node2])
    
    graph = dgl.heterograph(graph)

    dgl.save_graphs("../data/go_cat.bin", graph)


    logging.debug(f"Type of node_idx: {type(node_idx)}")
    node_idx = {str(v.toStringID()): k for k, v in enumerate(go_classes)}
    
    with open("../data/nodes_cat.pkl", "wb") as pkl_file:
        pkl.dump(node_idx, pkl_file)
    # axioms = ontology.getAxioms(list(go_classes)[17])
    # print(axioms)

    # print("\n\n\n")
    # print(list(axioms)[2].getClassExpressionsAsList())
    # print(type(list(axioms)[2].getClassExpressionsAsList()[0].toStringID()))
    # print(len(axioms))
    # print(len(classes))
    # print(len(individuals))


def processAxioms(go_class):
    axioms = ontology.getAxioms(go_class)
    edges = []
    for axiom in axioms:
        axiomType = axiom.getAxiomType().getName()

        if axiomType == "EquivalentClasses":
            expressions = axiom.getClassExpressionsAsList()
            expressions.pop(0) # Remove the first element, which is the go class

            for expr in expressions:
                new_edges = processExpressions(go_class, expr)
            edges += new_edges
        elif axiomType == "SubClassOf":
            edges += processSubClassOfAxiom(axiom)

        else:
            logging.info(f"axiom type missing: {axiomType}")

    return edges


def processExpressions(go_class, expr):
    exprType = expr.getClassExpressionType().getName()
    edges = []
    if exprType == "ObjectIntersectionOf":
        operands = expr.getOperands()
        for op in operands:
            opType = op.getClassExpressionType().getName()
            if opType == "Class":
                edges.append((go_class, "projects", op))
            elif opType == "ObjectSomeValuesFrom":
                relation = op.getProperty().toStringID()
                dst_class = op.getFiller()
                dst_type = dst_class.getClassExpressionType().getName()
                if dst_type == "Class":
                    edges.append((go_class, f"projects_{relation}", dst_class))
                else:
                    logging.info("Detected complex operand in intersection")

            else:
                logging.info(f"projection missing: {opType}")
    return edges

def processSubClassOfAxiom(axiom):
    subClass = axiom.getSubClass()
    superClass = axiom.getSuperClass()

    subClassType = subClass.getClassExpressionType().getName()
    superClassType = superClass.getClassExpressionType().getName()

    if subClassType == "Class" and superClassType == "Class":
        edges = [(subClass, "is_a", superClass)]
    else:
        edges = []
        logging.info("Detected complex subclass or superclass in subClassOf")

    return edges

if __name__ == '__main__':
    main()

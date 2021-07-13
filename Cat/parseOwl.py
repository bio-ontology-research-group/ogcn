
import jpype
import jpype.imports
import os
import logging

logging.basicConfig(level=logging.DEBUG)

jars_dir = "../jars/"
jars = f'{str.join(":", [jars_dir + name for name in os.listdir(jars_dir)])}'

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(), "-ea",
        "-Djava.class.path=" + jars,
        convertStrings=False)


#JPype imports
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
        node2 = node_idx[go_class_2]

        graph[key].append([node1, node2])
    
    graph = dgl.heterograph(graph)

    dgl.save_graphs("../data/go_cat.bin", graph)
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
    return edges

if __name__ == '__main__':
    main()

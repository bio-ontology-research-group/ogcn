

import os
import pickle as pkl
import logging


logging.basicConfig(level=logging.DEBUG)


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

    global data_factory
    data_factory = ont_manager.getOWLDataFactory()

    global ontology 
    ontology = ont_manager.loadOntologyFromOntologyDocument(File("../data/go-plus.owl"))
    
    axioms = ontology.getAxioms()

    imports = Imports.fromBoolean(False)
    go_classes = ontology.getClassesInSignature(imports)
    node_idx = {v: k for k, v in enumerate(go_classes)}

    global rel_counter
    rel_counter = 0
    edges = []
    for go_class in go_classes:
        edges.append((go_class, "id", go_class))
        new_edges = processAxioms(go_class)
        edges += new_edges

    logging.info(f"Number of edges: {len(edges)}")

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
    
    graph = {k: v for k, v in graph.items() if len(v) > 1000}

    rels = {k: len(v) for k, v in graph.items()}

    logging.info(f"Number of nodes: {len(go_classes)}")
    logging.info(f"Edges in the graph:\n{rels}")

    graph = dgl.heterograph(graph)

    dgl.save_graphs("../data/go_cat.bin", graph)


    logging.debug(f"Type of node_idx: {type(node_idx)}")
    node_idx = {prettyFormat(v): k for k, v in enumerate(go_classes)}
    
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

def prettyFormat(go_class):
    go_class_str = str(go_class.toStringID()).split('/')[-1]
    go_class_str = go_class_str.replace("_", ":",1)
    return go_class_str

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
        elif axiomType == "DisjointClasses":
            edges += processDisjointness(axiom)
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
                relation, dst_class = processObjectSomeValuesFrom(op)
                dst_type = dst_class.getClassExpressionType().getName()
                if dst_type == "Class":
                    edges.append((go_class, "projects_" + relation, dst_class))
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

    edges = []
    if subClassType == "Class" and superClassType == "Class":
        edges = [(subClass, "is_a", superClass)]
    elif subClassType == "Class" and superClassType == "ObjectSomeValuesFrom":
        relation, dst_class = processObjectSomeValuesFrom(superClass)
        dst_type = dst_class.getClassExpressionType().getName()
        if dst_type == "Class":
            edges.append((subClass, f"{relation}", dst_class))
        else:
            logging.info("Detected complex operand in SubClassOf axiom")
    else:
        edges = []
        logging.info(f"Detected complex subclass or superclass in subClassOf: {subClassType}, {superClassType}")

    return edges

def processDisjointness(axiom):
    exprs = axiom.getClassExpressionsAsList()

    edges = []

    if len(exprs) > 2:
        logging.info("More than two operands in Disjoint axiom")
    else:
        src = exprs[0]
        dst = exprs[1]

        srcType = src.getClassExpressionType().getName()
        dstType = dst.getClassExpressionType().getName()
        if srcType == "Class" and dstType == "Class":
            edges.append((src, "disjointWith", dst))
            edges.append((dst, "disjointWith", src))
        else:
            logging.info(f"Detected complex nodes in disjointWith: {srcType}, {dstType}")

    return edges

def processObjectSomeValuesFrom(expr):
    global rel_counter
    relation = expr.getProperty() #.toStringID()

    rel = None
    #to get name of relation
    for annot in ontology.getAnnotationAssertionAxioms(relation.getIRI()):
        if annot.getProperty() == data_factory.getRDFSLabel() :
            rel = str(annot.getValue()).strip('"').replace(' ', '_')
        
    if rel == None:
        logging.debug(f"relation: {rel_counter}")

        rel = "rel_" + str(rel_counter)
        rel_counter += 1
    dst_class = expr.getFiller()
    return rel, dst_class

if __name__ == '__main__':
    main()

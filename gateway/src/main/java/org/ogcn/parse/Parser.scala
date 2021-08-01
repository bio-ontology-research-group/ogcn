package org.ogcn.parse

// Java imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import java.io.File


import collection.JavaConverters._

import org.ogcn.parse.Types._


class Parser(var ont_path:String){

    val ont_manager = OWLManager.createOWLOntologyManager()
    val data_factory = ont_manager.getOWLDataFactory()
    val ontology = ont_manager.loadOntologyFromOntologyDocument(new File(ont_path))

    val axioms = ontology.getAxioms()
    val imports = Imports.fromBoolean(false)

    val go_classes = ontology.getClassesInSignature(imports).asScala.toList

    val node_idx = go_classes.zipWithIndex.map{case (v, i) => (i,goClassToStr(v))}.toMap

    val id_edges = go_classes.map(((x: String) => new Edge(x, "id", x)) compose goClassToStr)


    val edges = go_classes.foldLeft(id_edges){(acc, x) => acc ::: processGOClass(x)}



    def goClassToStr(goClass: OWLClass) = goClass.toStringID.split("/").last.replace("_", ":")
    
    def processGOClass(go_class: OWLClass): List[Edge] = List[Edge]()
}
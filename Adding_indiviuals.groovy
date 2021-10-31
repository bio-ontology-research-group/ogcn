//-----------------------------------------------------------
// Sarah M. Algamdi
//-----------------------------------------------------------
// This code is used to add indivisuals to classes in an ontology with types, 
// outputs ontology with the inserted indiviguals
//
//inputs:
//  ontology_path
//  input_csv: a comma seperated value file with two columns(indivisual_id , class_url)
//  outpt_ontology_path
// to run:
// AddingIndivisuals.groovy ontology_path input_csv outpt_ontology_path
//-----------------------------------------------------------



@Grapes([
@Grab(group='org.slf4j', module='slf4j-simple', version='1.6.1'),
@Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.1'),
@Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.1')
])

import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import java.io.File;


OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(new File(args[0]))
def onturi = "http://url/"

println("Ontology loaded ... ")

OWLDataFactory fac = manager.getOWLDataFactory()

map=[:]
new File(args[1]).splitEachLine(','){ line ->
    if (map.containsKey(line[0])){
        map[line[0]].add(line[1].replace("GO:","GO_"))
    }else{
        map[line[0]]=[line[1].replace("GO:","GO_")]
    }
}

OWLIndividual indivisual
OWLClassAssertionAxiom ax

println("Adding indiviuals ... ")

map.keySet().each{ g ->
    indivisual = fac.getOWLNamedIndividual(IRI.create(onturi + g));
    map[g].each{ annot ->
        println(g)
        println(annot)
        i_class = fac.getOWLClass(IRI.create(annot))
        ax = fac.getOWLClassAssertionAxiom(i_class, indivisual);
        manager.addAxiom(ont, ax)
    }
}

println("Saving new ontology ... ")

manager.saveOntology(ont, IRI.create((new File(args[2]).toURI())))

println("Done! ")
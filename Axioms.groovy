@Grapes([
  @Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.2'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.1.0'),
  @Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.1.0'),
  @GrabConfig(systemClassLoader=true)
  ])

import org.semanticweb.owlapi.model.parameters.*
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration
import org.semanticweb.elk.reasoner.config.*
import org.semanticweb.owlapi.manchestersyntax.renderer.*
import org.semanticweb.owlapi.formats.*
import java.util.*


OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(
  new File("data/go.owl"))
OWLDataFactory factory = manager.getOWLDataFactory()
ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
ElkReasonerFactory reasonerFactory = new ElkReasonerFactory()
OWLReasoner reasoner = reasonerFactory.createReasoner(ont, config)
reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
def renderer = new ManchesterOWLSyntaxOWLObjectRendererImpl()
def shortFormProvider = new SimpleShortFormProvider()

def getName = { cl ->
  def iri = cl.toString()
  def name = iri
  if (iri.startsWith("<http://purl.obolibrary.org/obo/")) {
    name = iri.substring(32, iri.length() - 1)
  } else if (iri.startsWith("<http://aber-owl.net/")) {
    name = iri.substring(21, iri.length() - 1)
  }
  return name
}

String base = "http://purl.obolibrary.org/obo/"
OWLObjectProperty hasFunc = factory.getOWLObjectProperty(IRI.create(base
                + "hasFunc"));

new File("data/9606.annotations.tsv").splitEachLine("\t") { items ->
    def protID = items[0]
    OWLClass prot = factory.getOWLClass(IRI.create(base + protID));
    for (int i = 1; i < items.size(); i++) {
        def goID = items[i].replaceAll(":", "_");
        OWLClass goClass = factory.getOWLClass(IRI.create(base + goID));
        OWLClassExpression hasFuncSomeGO = factory.getOWLObjectSomeValuesFrom(hasFunc,
                goClass);
        OWLSubClassOfAxiom newAxiom = factory.getOWLSubClassOfAxiom(prot, hasFuncSomeGO);
        manager.addAxiom(ont, newAxiom);
    }
}



File newOntFile = new File("data/go-prots-9606.owl");
manager.saveOntology(ont, IRI.create(newOntFile.toURI()))

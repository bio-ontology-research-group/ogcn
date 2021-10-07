//-----------------------------------------------------------
// Sarah M. Algamdi
//-----------------------------------------------------------
// This code is used to expand an exixting RDF graph with ontological nodes and edges
//-----------------------------------------------------------
//  

@Grapes([
    @Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.3'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5'),
    @Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.2.5'),
    @Grab(group='org.apache.jena', module='apache-jena-libs', version='3.9.0', type='pom')
  ])

import org.semanticweb.owlapi.model.parameters.*
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration
import org.semanticweb.elk.reasoner.config.*
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*
import org.semanticweb.owlapi.reasoner.structural.StructuralReasoner
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary;
import org.semanticweb.owlapi.model.*;
//import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;
import org.semanticweb.owlapi.reasoner.structural.*
//import org.apache.jena.rdf.*
import org.apache.jena.rdf.model.*

import org.apache.jena.util.*
import org.apache.jena.vocabulary.RDFS
import org.apache.jena.graph.Node
//import org.apache.jena.model.RDFNode
//import org.apache.jena.vocabulary.*

def cli = new CliBuilder()
cli.with {
usage: 'Self'
  h longOpt:'help', 'this information'
  o longOpt:'output', 'output RDF graph path',args:1//, required:true
  ont longOpt:'ontology', 'Ontology path required if the addition of the ontology , trasitivity over is_a or transitivity over parthood relation is selected',args:1, required:false
  g longOpt:'graph', 'input RDF file', args:1//, required:true
  annot longOpt:'annotaions', 'annotaion file path',args:1, required:false
  aa longOpt:'add_annotaions', 'add the annotaion t othe graph from an annotaion file'
  ao longOpt:'add_ontology', 'add ontology to the input RDF graph'
  atis longOpt:'add_t_is_a', 'add the deductive closure to the ontology is-a'
  atpo longOpt:'add_t_part_of', 'add trasitivity over part-of relation'
  a_box longOpt:'expand_a_box', 'When adding axioms of the deductive closure, this option expands the A-Box'
  t_box longOpt:'expand_t_box', 'When adding axioms of the deductive closure, this option expands the T-Box'
  sl  longOpt:'self_loops', 'Adding self-loops to reflexive relations'
  c longOpt:'classify', 'classify the ontology with ELK'
}
def opt = cli.parse(args)
if( !opt ) {
  println("no argumants provided! use -h for help")
  return
}
if( opt.h ) {
    cli.usage()
    return
}

if ((opt.ao || opt.atis || opt.atpo) && !opt.ont) {
  println("please provide an ontology")
  return
}

if (opt.aa && !opt.annot) {
  println("please provide an annotaion file")
  return
}

if (!opt.a_box && !opt.t_box) {
  println("choose either to expand the A-Box or the T-box")
  return
}


if (opt.atpo){
  println("Warning !! (this code works for the part-of relation in GO if you are using another ontology please check the url of part-of relationa and fix it )")
}


classify = false
if(opt.c){
  classify = true
}

def f = File.createTempFile("temp",".tmp")

if (classify) {
  OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
  def oset = new LinkedHashSet()
  oset.add(manager.loadOntologyFromOntologyDocument(new File(opt.ont)))

  OWLOntology ont = manager.createOntology(IRI.create("http://aber-owl.net/rdfwalker/t.owl"),oset)
  OWLDataFactory fac = manager.getOWLDataFactory()
  ConsoleProgressMonitor progressMonitor = new ConsoleProgressMonitor()
  OWLReasonerConfiguration config = new SimpleConfiguration(progressMonitor)
  ElkReasonerFactory f1 = new ElkReasonerFactory()
  OWLReasoner reasoner = f1.createReasoner(ont,config)
  def cc = 0
  new InferredClassAssertionAxiomGenerator().createAxioms(fac, reasoner).each { ax ->
    manager.addAxiom(ont, ax)
    cc += 1
  }
  manager.saveOntology(ont, IRI.create(f.toURI()))
  println "$cc axioms inferred."
}

def filename = null
if (classify) {
  filename = f.toURI()
} else {
  filename = opt.ont
}

Model model = ModelFactory.createDefaultModel()
InputStream infile = FileManager.get().open( filename.toString() )
def format = "RDF/XML"
model.read(infile, null, format)

/*

if(opt.g){
  Model model2 = ModelFactory.createDefaultModel()
  infile = FileManager.get().open( opt.g.toString() )
  model.read(infile, null, "Turtle")
  Model model1 = model.union(model2)
  model = model1
  //outstreem = new FileOutputStream("added_ontology_and_graph.ttl");
  //model.write(outstreem, "Turtle")
}


*/




url_for_wf = "http://ogcn/"
model.setNsPrefix("ogcn",url_for_wf)

//save in temp file
//OutputStream outstreem = new FileOutputStream("temp.rdf");
//model.write(outstreem, format)
rdf_objects=[:]


if(opt.aa){
  manager = OWLManager.createOWLOntologyManager()
  ont = manager.loadOntologyFromOntologyDocument(new File(opt.ont))
  fac = manager.getOWLDataFactory()
  progressMonitor = new ConsoleProgressMonitor()
  config = new SimpleConfiguration(progressMonitor)
  f1 = new ElkReasonerFactory()
  reasoner = f1.createReasoner(ont,config) 
  Property is_a_property = RDFS.subClassOf
  Property part_of_property = model.createProperty("http://purl.obolibrary.org/obo/BFO_0000050")


  Property has_annotation = model.createProperty(url_for_wf+"has_annotaion")
  new File(opt.annot).splitEachLine('\t'){line ->

      class_id = line[0].replaceAll("\\s","")
      class_url = "http://purl.obolibrary.org/obo/"+class_id.replaceAll(":","_")
      class_ex = fac.getOWLClass(IRI.create(class_url))
      entity = url_for_wf+line[1].replaceAll("\\s","")
      Resource cl1 = model.createResource(class_url)
      Resource entity_resource = model.createResource(entity)
      model.add(entity_resource, has_annotation,cl1)
      println("edge added for normal annotaion")


      if(opt.a_box && opt.atis)
      {
        reasoner.getSuperClasses(class_ex, false).getFlattened().each { n->
        uri = n.toString().replaceAll(">","").replaceAll("<","")
        label= n.toString().replaceAll(">","").replaceAll("<","")
        if(label.indexOf("Thing")==-1){
          Resource cl2 = model.createResource(n.toString().replaceAll(">","").replaceAll("<",""))
          Resource entity_resource = model.createResource(annotaion)
          model.add(entity_resource, has_annotation,cl2)
          println("edge added for A-Box deductive closure over is-a relation")
        }
      }


      }

      if(opt.a_box && opt.atpo)
      {
        part_of = fac.getOWLObjectProperty(IRI.create("http://purl.obolibrary.org/obo/BFO_0000050")) 
        class_ex = fac.getOWLClass(IRI.create(class_url))
        //part_of_class = fac.getOWLObjectSomeValuesFrom(part_of,class_ex)



        reasoner.getSuperClasses(class_ex, false).getFlattened().each {
        superClass -> 

        if (superClass.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM && superClass.getProperty()?.toString() == part_of){
          if( superClass.getFiller().getClassExpressionType() == ClassExpressionType.OWL_CLASS){
            obj = id(superClass.getFiller().toString())
            Resource cl2 = model.createResource(obj.replaceAll(">","").replaceAll("<",""))
            Resource entity_resource = model.createResource(annotaion)
            model.add(entity_resource, has_annotation,cl2)
            println("edge added for A-Box deductive closure over part-of relation")

          }              
          }

        }

      }

    }
    //outstreem = new FileOutputStream("added_annotations.ttl");
    //model.write(outstreem, "Turtle")
}





// add_t_is_a

if (opt.atis && opt.t_box ){
  manager = OWLManager.createOWLOntologyManager()
  ont = manager.loadOntologyFromOntologyDocument(new File(opt.ont))
  fac = manager.getOWLDataFactory()
  progressMonitor = new ConsoleProgressMonitor()
  config = new SimpleConfiguration(progressMonitor)
  f1 = new ElkReasonerFactory()
  reasoner = f1.createReasoner(ont,config) 

  Property is_a_property = RDFS.subClassOf

  ont.getClassesInSignature(true).each{ cl ->
      uri = cl.toString().replaceAll(">","").replaceAll("<","")
      Resource cl1 = model.createResource(cl.toString().replaceAll(">","").replaceAll("<",""))

      reasoner.getSuperClasses(cl, false).getFlattened().each { n->
        uri = n.toString().replaceAll(">","").replaceAll("<","")

        println(n.toString())

        label= n.toString().replaceAll(">","").replaceAll("<","")
        if(label.indexOf("Thing")==-1){

          Resource cl2 = model.createResource(n.toString().replaceAll(">","").replaceAll("<",""))

          model.add(cl1,is_a_property,cl2)
          println("edge added for T-Box deductive closure over is-a relation")

        }
         }
  }

  //outstreem = new FileOutputStream("added_deductive_closer.ttl");
  //model.write(outstreem, "Turtle")

}

  def add_parents(ls, go){

    if(!map_for_part_of.containsKey(go)){
      return 
    }else{
      map_for_part_of[go].each{cl ->
        ls.add(cl)
        add_parents(ls, cl)
        return }
    }
  }


// add_t_part_of

if (opt.atpo && opt.t_box){
  def id = {cl->cl
    return cl.replaceAll(">","").replaceAll("<","")
  }

  manager = OWLManager.createOWLOntologyManager()
  ont = manager.loadOntologyFromOntologyDocument(new File(opt.ont))
  map_for_part_of = [:]
  part_of = "<http://purl.obolibrary.org/obo/BFO_0000050>"
  Property part_of_property = model.createProperty("http://purl.obolibrary.org/obo/BFO_0000050")
  ont.getClassesInSignature(true).each { cl ->
    
    EntitySearcher.getSuperClasses(cl, ont).each {
    superClass -> 

    if (superClass.getClassExpressionType() == ClassExpressionType.OBJECT_SOME_VALUES_FROM && superClass.getProperty()?.toString() == part_of){

        obj = id(superClass.getFiller().toString())
        subj = id(cl.toString())

        if(!map_for_part_of.containsKey(subj)){
              map_for_part_of[subj] = []
            }
            map_for_part_of[subj].add(obj)        
      }

    }
  
  }




  map_for_non_direct_part_of = [:]

  map_for_part_of.each{go,value ->
    map_for_non_direct_part_of[go]= []
    add_parents(map_for_non_direct_part_of[go],go)       
  }


  map_for_non_direct_part_of.each{go,value ->
    value.each{super_go ->
      Resource cl1 = model.createResource(go)
      Resource cl2 = model.createResource(super_go)
      model.add(cl1,part_of_property,cl2)
      println("edge added for T-Box deductive closure over part-of relation")
      println(go)
      println(super_go)

    }
  }

}

if (opt.sl){


  
}


// saving the final graph
output = "output.ttl"
if(opt.o)
{
  output=opt.o
}

outstreem = new FileOutputStream(output)
model.write(outstreem, "Turtle")
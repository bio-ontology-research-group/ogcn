package org.ogcn.parse

import org.semanticweb.owlapi.model._

import uk.ac.manchester.cs.owl.owlapi.OWLClassImpl

object Types {

    val Top = new OWLClassImpl(IRI.create("http://www.w3.org/2002/07/owl#Thing")).asOWLClass
    val Bottom = new OWLClassImpl(IRI.create("http://www.w3.org/2002/07/owl#Nothing")).asOWLClass


    

    type GOClass = String
    type Relation = String


    sealed trait QuantifiedExpression {
        def getProperty(): OWLObjectPropertyExpression
        def getFiller(): OWLClassExpression 
    }
   
    case class Universal(val expression: OWLObjectAllValuesFrom) extends QuantifiedExpression{
        def getProperty() = expression.getProperty
        def getFiller() = expression.getFiller
    }
    case class Existential(val expression: OWLObjectSomeValuesFrom) extends QuantifiedExpression{
        def getProperty() = expression.getProperty
        def getFiller() = expression.getFiller
    }
    case class MinCardinality(val expression: OWLObjectMinCardinality) extends QuantifiedExpression{
        def getProperty() = expression.getProperty
        def getFiller() = expression.getFiller
    }


    sealed trait Expression
    case object GOClass extends Expression
    case class ObjectSomeValuesFrom(rel: Relation, expr: Expression) extends Expression

    sealed trait Axiom
    case class SubClassOf(subClass: GOClass, superClass: Expression) extends Axiom
    case class Equivalent(leftSide: GOClass, rightSide: List[Expression])


    class Edge(val src:GOClass, val rel:Relation, val dst:GOClass){
        def this(src: OWLClass, rel: Relation, dst: OWLClass) = this(goClassToStr(src), rel, goClassToStr(dst))
        def this(src: String, rel: Relation, dst: OWLClass) = this(src, rel, goClassToStr(dst))
        def this(src: OWLClass, rel: Relation, dst: String) = this(goClassToStr(src), rel, dst)
    }

    def goClassToStr(goClass: OWLClass) = goClass.toStringID.split("/").last.replace("_", ":")

    def getNodes(edges: List[Edge]) = {
        
        val edges_sc = edges
        val srcs = edges_sc.map((e) => e.src)
        val dsts = edges_sc.map((e) => e.dst)

        (srcs ::: dsts).toSet
    }

}

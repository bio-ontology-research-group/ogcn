package org.ogcn.parse

import org.semanticweb.owlapi.model._


object Types {



    type GOClass = String
    type Relation = String

    sealed trait Expression
    case object GOClass extends Expression
    case class ObjectSomeValuesFrom(rel: Relation, expr: Expression) extends Expression

    sealed trait Axiom
    case class SubClassOf(subClass: GOClass, superClass: Expression) extends Axiom
    case class Equivalent(leftSide: GOClass, rightSide: List[Expression])


    class Edge(src:GOClass, rel:Relation, dst:GOClass){
        def this(src: OWLClass, rel: Relation, dst: OWLClass) = this(goClassToStr(src), rel, goClassToStr(dst))
        def this(src: String, rel: Relation, dst: OWLClass) = this(src, rel, goClassToStr(dst))
        def this(src: OWLClass, rel: Relation, dst: String) = this(goClassToStr(src), rel, dst)
    }

    def goClassToStr(goClass: OWLClass) = goClass.toStringID.split("/").last.replace("_", ":")
}

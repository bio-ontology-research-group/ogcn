package org.ogcn.parse

object Types {



    type GOClass = String
    type Relation = String

    sealed trait Expression
    case object GOClass extends Expression
    case class ObjectSomeValuesFrom(rel: Relation, expr: Expression) extends Expression

    sealed trait Axiom
    case class SubClassOf(subClass: GOClass, superClass: Expression) extends Axiom
    case class Equivalent(leftSide: GOClass, rightSide: List[Expression])


    class Edge(src:GOClass, rel:Relation, dst:GOClass)

}

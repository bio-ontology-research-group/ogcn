package org.ogcn.parse

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports



// Java imports
import java.io.File


import collection.JavaConverters._

import org.ogcn.parse.Types._


class Parser(var ont_path: String) {

    private val ont_manager = OWLManager.createOWLOntologyManager()
    private val ontology = ont_manager.loadOntologyFromOntologyDocument(new File(ont_path))
    private val data_factory = ont_manager.getOWLDataFactory()

    var rel_counter = 0

    def parse = {
           
        val axioms = ontology.getAxioms()
        val imports = Imports.fromBoolean(false)
        val go_classes = ontology.getClassesInSignature(imports).asScala.toList


       
        
        val edges = go_classes.foldLeft(List[Edge]()){(acc, x) => acc ::: processGOClass(x)}

        val nodes = getNodes(edges)

        val id_edges = nodes.map((x) => new Edge(x, "id", x)).toList


        (id_edges ::: edges).asJava
    }

    

   
    
    def processGOClass(go_class: OWLClass): List[Edge] = {
        val axioms = ontology.getAxioms(go_class).asScala.toList

        val edges = axioms.flatMap(parseAxiom(go_class, _: OWLClassAxiom))
        edges
    }

    def parseAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Edge] = {
        val axiomType = axiom.getAxiomType().getName()
        axiomType match {
            case "EquivalentClasses" => {
                var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala.toList
                ax.tail.flatMap(parseEquivClass(go_class, _: OWLClassExpression))
            }
            case "SubClassOf" => {
                var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
                parseSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
            }
            case "DisjointClasses" => {
                var ax = axiom.asInstanceOf[OWLDisjointClassesAxiom].getClassExpressionsAsList.asScala.toList
                ax.tail.flatMap(parseDisjointnessAxiom(go_class, _: OWLClassExpression))
            }
            case _ =>  throw new Exception(s"Not parsing axiom $axiomType")
        }
    }


    /////////////////////////////////////////////
    def parseEquivClass(go_class: OWLClass, rightSideExpr: OWLClassExpression) = {
        val exprType = rightSideExpr.getClassExpressionType().getName()

        exprType match {
            case "ObjectIntersectionOf" =>  {
                var expr = rightSideExpr.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList
                expr.map(parseIntersection(go_class, _: OWLClassExpression))
                }
            case _ =>  throw new Exception(s"Not parsing EquivalentClass rigth side $exprType")
        }

    }

    def parseDisjointnessAxiom(go_class: OWLClass, rightSideExpr: OWLClassExpression) = {
        val exprType = rightSideExpr.getClassExpressionType().getName()

        val left_proj = new Edge("Bottom", "projects", go_class)

        exprType match {
            case "Class" => {
                val expr = rightSideExpr.asInstanceOf[OWLClass]
                val right_proj = new Edge("Bottom", "projects", expr)
                
                left_proj :: right_proj :: Nil
            }
            case _ => throw new Exception(s"Not parsing Disjointness rigth side $exprType")
        }
    }

    def parseSubClassAxiom(go_class: OWLClass, superClass: OWLClassExpression) = {
        val superClassType = superClass.getClassExpressionType.getName

        val neg_sub = new Edge(s"Not_${goClassToStr(go_class)}", "negate", go_class)

        val injection_sub = parseUnion(Top, go_class, "SubClass") // new Edge(go_class, "injects", "Top")

        val injection_super = parseUnion(Top, superClass, "SubClass")

        neg_sub :: injection_sub :: injection_super :: Nil

    }


    /////////////////////////////////////////////
    def parseIntersection(go_class: OWLClass, projected_expr: OWLClassExpression) = {
        val exprType = projected_expr.getClassExpressionType.getName

        exprType match {
            case "Class" => {
                val proj_class = projected_expr.asInstanceOf[OWLClass]
                new Edge(go_class, "projects", proj_class)
                }
            case "ObjectSomeValuesFrom" => {
                val proj_class = projected_expr.asInstanceOf[OWLObjectSomeValuesFrom]
                
                val(rel, dst_class) = parseObjectSomeValuesFrom(proj_class) 

                val dst_type = dst_class.getClassExpressionType.getName
                dst_type match {
                    case "Class" => {
                        val dst = dst_class.asInstanceOf[OWLClass]
                        new Edge(go_class, "projects_" + rel, dst)
                        }
                    case _ =>  throw new Exception(s"Not parsing Filler in ObjectSomeValuesFrom(Intersection) $dst_type")
                }
            } 
            case _ =>  throw new Exception(s"Not parsing Intersection operand $exprType")
        }

    }

    def parseUnion(go_class: OWLClass, injected_expr: OWLClassExpression, origin: String = "Union") = {
        val exprType = injected_expr.getClassExpressionType.getName

        exprType match {
            case "Class" => {
                val inj_class = injected_expr.asInstanceOf[OWLClass]
                new Edge(inj_class, "injects", go_class)
                }
            case "ObjectSomeValuesFrom" => {
                val proj_class = injected_expr.asInstanceOf[OWLObjectSomeValuesFrom]
                
                val(rel, src_class) = parseObjectSomeValuesFrom(proj_class, true) 

                val src_type = src_class.getClassExpressionType.getName
                src_type match {
                    case "Class" => {
                        val src = src_class.asInstanceOf[OWLClass]
                        new Edge(src, "injects_" + rel, go_class)
                        }
                    case _ =>  throw new Exception(s"Not parsing Filler in ObjectSomeValuesFrom(Intersection) $src_type")
                }
            } 
            case _ =>  throw new Exception(s"Not parsing Union ($origin) operand $exprType")
        }

    }

    def parseObjectSomeValuesFrom(expr: OWLObjectSomeValuesFrom, inverse: Boolean = false) = {
        
        var relation = expr.getProperty.asInstanceOf[OWLObjectProperty]

        if (inverse) {
            var inv_relation = relation.getInverseProperty
            if (!inv_relation.isAnonymous){
                relation = relation.asOWLObjectProperty
            }
        }


        val rel_annots = ontology.getAnnotationAssertionAxioms(relation.getIRI()).asScala.toList

        val rel = rel_annots find (x => x.getProperty() == data_factory.getRDFSLabel()) match {
            case Some(r) => r.getValue().toString.replace("\"", "").replace(" ", "_")
            case None => {
                rel_counter = rel_counter + 1
                "rel" + (rel_counter)
            }
        }
        val dst_class = expr.getFiller()

        val dst_type = dst_class.getClassExpressionType.getName

        if (inverse){
            var inv_relation = relation.getInverseProperty
            if (!inv_relation.isAnonymous){
                (rel, dst_class)
            }else{
                ("inv_" + rel, dst_class)
            }
        }else{
            (rel, dst_class)
        }
        
    }
}
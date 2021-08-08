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
                ax.filter(_ != go_class).flatMap(parseEquivClassAxiom(go_class, _: OWLClassExpression))
            }
            case "SubClassOf" => {
                var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
                parseSubClassAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass)
            }
            case "DisjointClasses" => {
                var ax = axiom.asInstanceOf[OWLDisjointClassesAxiom].getClassExpressionsAsList.asScala.toList
                ax.filter(_ != go_class).flatMap(parseDisjointnessAxiom(go_class, _: OWLClassExpression))
            }
            case _ =>  throw new Exception(s"Not parsing axiom $axiomType")
        }
    }


    /////////////////////////////////////////////
    def parseEquivClassAxiom(go_class: OWLClass, rightSideExpr: OWLClassExpression) = {
        val exprType = rightSideExpr.getClassExpressionType().getName()

        exprType match {
            case "ObjectIntersectionOf" =>  {
                var expr = rightSideExpr.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList

                
                expr.map(parseIntersection(go_class, _: OWLClassExpression))
                }
            case "ObjectUnionOf" => {
                var expr = rightSideExpr.asInstanceOf[OWLObjectUnionOf].getOperands.asScala.toList
                expr.flatMap(parseUnion(go_class, _: OWLClassExpression))
            }
            case _ =>  throw new Exception(s"Not parsing EquivalentClass rigth side $exprType")
        }

    }

    def parseDisjointnessAxiom(go_class: OWLClass, rightSideExpr: OWLClassExpression) = {
        val exprType = rightSideExpr.getClassExpressionType().getName()

        val left_proj = projectionMorphism(Bottom, go_class)

        val right_proj = parseIntersection(Bottom, rightSideExpr, None, origin = "Disjointness")

        left_proj :: right_proj :: Nil
    }

    def parseSubClassAxiom(go_class: OWLClass, superClass: OWLClassExpression): List[Edge] = {

        // println(s"$go_class\n$superClass")
        val neg_sub = negationMorphism(go_class)

        val injection_sub = parseUnion(Top, go_class, origin="SubClass") // new Edge(go_class, "injects", "Top")

        val injections_super = parseUnion(Top, superClass, origin="SubClass")

        neg_sub :: injection_sub ::: injections_super

    }


    /////////////////////////////////////////////
    def parseIntersection(go_class: OWLClass, projected_expr: OWLClassExpression, prevRel: Option[String] = None, origin: String = "Equiv") : Edge = {
        val exprType = projected_expr.getClassExpressionType.getName

        exprType match {
            case "Class" => projectionMorphism(go_class, projected_expr)
            case "ObjectSomeValuesFrom" => {
                val proj_class = projected_expr.asInstanceOf[OWLObjectSomeValuesFrom]
                
                val(rel, dst_class) = parseQuantifiedExpression(Existential(proj_class)) 

                val dst_type = dst_class.getClassExpressionType.getName
                dst_type match {
                    case "Class" => projectionMorphism(go_class, dst_class, Some(rel))
                    case "ObjectSomeValuesFrom" => {
                        prevRel match {
                            case None => parseIntersection(go_class, dst_class, Some(rel))
                            case Some(r) => parseIntersection(go_class, dst_class, Some(r+"_"+rel))
                        }
                    }
                    case _ =>  throw new Exception(s"Not parsing Filler in ObjectSomeValuesFrom(Intersection) $dst_type\n$go_class\n$projected_expr")
                }
            }    
            case _ =>  throw new Exception(s"Not parsing Intersection ($origin) operand $exprType")
        }

    }

    def parseUnion(go_class: OWLClass, injected_expr: OWLClassExpression, prevRel: Option[String] = None, origin: String = "Union"): List[Edge] = {
        val exprType = injected_expr.getClassExpressionType.getName

        exprType match {
            case "Class" => injectionMorphism(injected_expr, go_class, prevRel) :: Nil

            case "ObjectComplementOf" => {
                val operand = injected_expr.asInstanceOf[OWLObjectComplementOf].getOperand
                val operandType = operand.getClassExpressionType.getName

                operandType match {
                    case "Class" => {
                        val neg = negationMorphism(operand)
                        val injection = parseUnion(go_class, operand, prevRel,"rec union OC")
                        neg :: injection
                    }
                    case _ => {
                        val injected_NNF = injected_expr.getNNF
                        parseUnion(go_class, injected_NNF, prevRel)
                    }
                }
            }

            case "ObjectSomeValuesFrom" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectSomeValuesFrom]
                
                val(rel, src_class) = parseQuantifiedExpression(Existential(inj_class), true) 

                val src_type = src_class.getClassExpressionType.getName

                prevRel match {
                    case None => parseUnion(go_class, src_class, Some(rel), "rec union OSV") // simple case

                    case Some(r) => parseUnion(go_class, src_class, Some(r + "_" + rel))

                    case _ => throw new Exception(s"Complex structure in ObjectSomeValuesFrom $src_type\n$origin\n$go_class\n$injected_expr")

                }
            }
         
            case "ObjectAllValuesFrom" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectAllValuesFrom]
                
                val(rel, src_class) = parseQuantifiedExpression(Universal(inj_class), true) 

                val src_type = src_class.getClassExpressionType.getName
                
                prevRel match {
                    case None => parseUnion(go_class, src_class, Some(rel), "rec union OAV") // simple case

                    case _ => throw new Exception(s"Complex structure in ObjectAllValuesFrom $src_type")

                }
            }

            case "ObjectMinCardinality" => {
                val inj_class = injected_expr.asInstanceOf[OWLObjectMinCardinality]
                
                val(rel, src_class) = parseQuantifiedExpression(MinCardinality(inj_class), true) 

                val src_type = src_class.getClassExpressionType.getName
                
                prevRel match {
                    case None => parseUnion(go_class, src_class, Some(rel), "rec union OMC") // simple case

                    case _ => throw new Exception(s"Complex structure in ObjectMinCardinality $src_type")

                }
            }

            
            // case "ObjectIntersectionOf" => {
            //     val inj_class = injected_expr.asInstanceOf[OWLObjectIntersectionOf]
            //     val exactCardinality = checkExactCardinality(inj_class)
                
            //     exactCardinality match {
            //         case None => throw new Exception(s"Not parsed complex intersection in union: $origin")
            //         case Some(expr) => parseUnion(go_class, expr, prevRel, "exactCardinality")
            //     }
            // }

            case _ =>  throw new Exception(s"Not parsing Union ($origin) operand $exprType\n$go_class")
        }

    }

    def parseQuantifiedExpression(expr: QuantifiedExpression, inverse: Boolean = false) = {
        
        var relation = expr.getProperty.asInstanceOf[OWLObjectProperty]

        val rel = getRelationName(relation, inverse)

        val dst_class = expr.getFiller

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

    def getRelationName(relation: OWLObjectProperty, inverse: Boolean = false) = {
        
        var relat = relation 
        if (inverse) {
            val inv_relation = relation.getInverseProperty
            if (!inv_relation.isAnonymous){
                relat = inv_relation.asOWLObjectProperty
            }
            
        }

        val rel_annots = ontology.getAnnotationAssertionAxioms(relat.getIRI()).asScala.toList

        val rel = rel_annots find (x => x.getProperty() == data_factory.getRDFSLabel()) match {
            case Some(r) => r.getValue().toString.replace("\"", "").replace(" ", "_")
            case None => {
                rel_counter = rel_counter + 1
                "rel" + (rel_counter)
            }
        }

        rel
    }

    def checkExactCardinality(expr: OWLObjectIntersectionOf) = {
        val operands = expr.getOperands
        None
    }

    ///////////////////////////////////////

    def negationMorphism(go_class: OWLClassExpression) = {
        val go_class_OWLClass = go_class.asInstanceOf[OWLClass]
        new Edge(s"Not_${goClassToStr(go_class_OWLClass)}", "negate", go_class_OWLClass)
    }

    def injectionMorphism(src: OWLClassExpression, dst: OWLClassExpression, rel: Option[String] = None) = {
        val src_OWLClass = src.asInstanceOf[OWLClass]
        val dst_OWLClass = dst.asInstanceOf[OWLClass]
        rel match {
            case Some(r) => new Edge(src_OWLClass, "injects_" + r, dst_OWLClass)
            case None => new Edge(src_OWLClass, "injects", dst_OWLClass)
        }
    }

    def projectionMorphism(src: OWLClassExpression, dst: OWLClassExpression, rel: Option[String] = None) = {
        val src_OWLClass = src.asInstanceOf[OWLClass]
        val dst_OWLClass = dst.asInstanceOf[OWLClass]
        rel match {
            case Some(r) => new Edge(src_OWLClass, "projects_" + r, dst_OWLClass)
            case None => new Edge(src_OWLClass, "projects", dst_OWLClass)
        }
    }
}
import numpy as np
import projective as pr

def general_loss(head, rel, tail):
    head = pr.homogenize(head)
    head = pr.transform(rel,head)
    head = pr.dehomogenize(head)

    loss = np.real(head.dot(tail))
    return loss

def neg_loss(head, rel, tail):
    

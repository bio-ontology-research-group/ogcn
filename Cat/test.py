import unittest
import numpy as np
from embeddings.projective import transform

class TestProjective(unittest.TestCase):
    def testTransform(self):
        self.assertTrue((transform(mat1, v1) == res1).all())
        self.assertTrue((transform(mat2, v2) == res2).all())
        self.assertTrue((transform(mat3, v3) == res3).all())




mat1 = np.array([[1,2],[3,4]])
v1 = np.array([2,3])
res1 = np.array([8, 18])

mat2 = np.array([[2,0],[1,1], [10, 1]])
v2 = np.array([0,3])
res2 = np.array([0, 3, 3])

mat3 = np.array([[1+1j,2+3j]])
v3 = np.array([2,3+3j])
res3 = np.array([-1+17j])


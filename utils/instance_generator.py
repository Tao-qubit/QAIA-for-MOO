import numpy as np
import time

import sys
sys.path.append('..')
from utils.functions import *

def generate_matrices(n=10,ds=0.5,seed=7):
	n = n
	ds = ds
	# a,b,c,d = 1,0,-2.35,1
	a ,b,c,d = 1,1,-2,-5
	np.random.seed(None)
	matrix0 = np.random.randint(0, 46, size=(n, n))
	matrix1 = generate_C2_from_C1(matrix0,r=46)

	matrix0 = (matrix0+5)*generate_mask(n,ds,seed=seed)
	matrix0 = (matrix0+matrix0.T)
	matrix1 = (matrix1+5)*generate_mask(n,ds,seed=seed)
	matrix1 = (matrix1+matrix1.T)

	X = matrix0 + matrix1
	matrix2 = -.5*matrix0-5*matrix1

	return matrix0,matrix1,matrix2
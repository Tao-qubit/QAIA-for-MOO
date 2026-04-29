import sys
import os
import torch
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
np.random.seed(666)

class SB():
    def __init__(self, A, h=0, tabu=None ,dt=1.,  n_iter=1000, xi=None, sk=False, batch_size=10, num_tabu=2, device='cpu'):
        self.N = A.shape[0]
        self.A = A.half()
        self.h = h
        self.tabu=tabu
        # self.A_sparse = csr_matrix(A)
        # The number of node
        self.batch_size = batch_size
        self.dt = dt
        self.M = 5
        self.n_iter = n_iter
        self.device = device
        self.p = np.linspace(0, 1,self.n_iter)
        self.dm = self.dt / self.M
        self.num_tabu = num_tabu
        self.sk = sk
        self.xi = xi
        if xi is None:
            if sk:
                self.xi = 0.7 * np.sqrt(self.N-1) / np.sqrt((self.A ** 2).sum())
            else:
                self.xi = 1 / np.abs(self.A.sum(axis=1)).max()
        
        
        self.initialize()

    def initialize(self):
        self.x = 0.01 * (torch.rand(self.N, self.batch_size, device=self.device,dtype=torch.float16)-0.5)
        self.y = 0.01 * (torch.rand(self.N, self.batch_size, device=self.device,dtype=torch.float16)-0.5)


    '''
    def calc_cut(self, x):
        sign = torch.sign(x)
        sign[sign==0] = 1.0
        cut = 0.25*(torch.sum((torch.mm(sign,self.A))*sign, dim=1))  - 0.25*self.asum
        return cut
    
    def energy_from_cut(self, cuts, offset=0):
        eng = -2*(cuts + 0.25*self.asum)+offset
        return eng
    
    def calc_energy(self, x):
        sign = torch.sign(x)
        sign[sign == 0] = 1.0
        energy = -0.5 * torch.mm(torch.mm(sign, self.A), sign.transpose(1,0))
        return energy
    '''
    

    
    def update(self):
        # iterate on the number of MVMs
        for i in range(self.n_iter):
            for j in range(self.M):
                self.y += -( self.x**2 + (1 - self.p[i]))*self.x*self.dm
                self.x += self.dm * self.y * self.delta

            self.y += self.xi * self.dt * self.A@self.x

    
    def update_b(self,beta=1,amp=0):
        # beta = beta
        for i in range(self.n_iter):
            self.y += (-(1 - self.p[i])*self.x + self.xi * (torch.sparse.mm(self.A, self.x)+self.h)+amp*torch.randn_like(self.x)) * self.dt
            self.x += self.dt * self.y 

            cond = torch.abs(self.x) > 1
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
            # self.y += -h* self.dt
            
        
    def update_d(self,beta=1,amp=0):
        # beta = beta
        for i in range(self.n_iter):
            self.y += (-(1 - self.p[i])*self.x + self.xi * (torch.sparse.mm(self.A, torch.sign(self.x))+self.h )+amp*torch.randn_like(self.x)) * self.dt
            self.x += self.dt * self.y 

            
            cond = torch.abs(self.x) > 1
            self.x = torch.where(cond, torch.sign(self.x), self.x)
            self.y = torch.where(cond, torch.zeros_like(self.y), self.y)
            

def read_gset(filename, negate=True):
    # read graph
    graph = pd.read_csv(filename, sep=' ')
    # the number of vertices
    n_v = int(graph.columns[0])
    # the number of edges
    n_e = int(graph.columns[1])

    assert n_e == graph.shape[0], 'The number of edges is not matched'

    G = csr_matrix((graph.iloc[:,-1], (graph.iloc[:, 0]-1, graph.iloc[:, 1]-1)), shape=(n_v, n_v))
    G = G+G.T       
    if negate:
        return -G
    else:
        return G



import numpy as np
import time
import torch
from itertools import combinations_with_replacement
import itertools
import moocore
from utils.functions import *
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3

if torch.cuda.is_available():
    import utils.NI_SB_gpu_half as tSB
else:
    import utils.NI_SB_gpu as tSB


def bf_samples(n=10):
	bool_vectors = list(itertools.product([False, True], repeat=n))
	bool_vectors = np.array(bool_vectors, dtype=bool)
	spin_vectors = 2*bool_vectors-1
	return spin_vectors

def tri_problem(n,seed=7):
	matrix0 = np.random.randint(0, 46, size=(n, n))
	matrix1 = generate_C2_from_C1(matrix0,r=46)
	matrix0 = (matrix0+5)*generate_mask(n,ds,seed=seed)
	matrix0 = (matrix0+matrix0.T)
	matrix1 = (matrix1+5)*generate_mask(n,ds,seed=seed)
	matrix1 = (matrix1+matrix1.T)
	matrix2 = -.5*matrix0-5*matrix1
	return matrix0,matrix1,matrix2


if __name__=="__main__":
	n2_tlist=[]
	n2_hvlist=[]
	n3_tlist=[]
	n3_hvlist=[]
	m_tlist=[]
	m_hvlist=[]
	k_tlist=[]
	k_hvlist=[]
	b_tlist=[]
	b_hvlist=[]
	d_tlist=[]
	d_hvlist=[]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	for n_loop in [10,20,100,200]:
		for d_loop in [0.5,1]:
			for reap in range(5):
				# print(n_loop, d_loop, reap)
				n = n_loop
				ds = d_loop
				# a ,b,c,d = 1,1,-2,-5
				np.random.seed(None)
				matrix0, matrix1, matrix2 = tri_problem(n)

				if n<25:
					spin_vectors = bf_samples(n)
				else:
					spin_vectors = np.random.randint(0, 2, size=(1000,n))*2-1

				obj0 = 0.5*(matrix0.sum() - (spin_vectors.dot(matrix0)*spin_vectors).sum(axis=1) )
				obj1 = 0.5*(matrix1.sum() - (spin_vectors.dot(matrix1)*spin_vectors).sum(axis=1) )
				obj2 = 0.5*(matrix2.sum() - (spin_vectors.dot(matrix2)*spin_vectors).sum(axis=1) )
				cost = np.array([obj0,obj1,obj2]).T

				answer = moocore.is_nondominated(cost, maximise=True, keep_weakly=False)
				# target = cost[answer].shape[0]
				low,upp = cost.min(axis=0),cost.max(axis=0)
				hv_max = moocore.hypervolume(-cost, ref=-np.array(low))
				print(hv_max)

				num_objectives = 3
				J0, J1, J2 = torch.from_numpy(matrix0).to(device, non_blocking=True),  torch.from_numpy(matrix1).to(device, non_blocking=True), torch.from_numpy(matrix2).to(device, non_blocking=True)


				start_ete_b = time.time()
				H_param = 9
				weights_array = generate_simplex_lattice_weights(n_obj=num_objectives, H=H_param)
				c_vectors = weights_array
				print(c_vectors.shape)
				# c_vectors = np.random.dirichlet(np.ones(num_objectives),size=100)
				J = torch.zeros((n*c_vectors.shape[0],n*c_vectors.shape[0])).to(device, non_blocking=True)
				if num_objectives==2:
					for j in range(c_vectors.shape[0]):
						J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1
				else:
					for j in range(c_vectors.shape[0]):
						J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1+c_vectors[j,2]*J2
				Ising_J = J
				xi = 1 / torch.abs(Ising_J.sum(axis=1)).max()
				Jm = Ising_J.to(device, non_blocking=True).float()
				Jm = Jm.to_sparse_csr()

				results = None
				n_iter = 50
				batch_size = 3000
				hv_ratio = 0
				while hv_ratio<0.001:
					s = tSB.SB(-Jm, n_iter=n_iter, xi=xi, dt=1, batch_size=batch_size,device=device)
					start = time.time()
					s.update_b(amp=0.15)
					end=time.time()

					rand_signs = 2.0 * torch.randint(0, 2, s.x.shape, device=s.x.device, dtype=s.x.dtype) - 1
					best_sample = torch.where(s.x == 0, rand_signs, s.x)
					best_sample = torch.sign(best_sample)
					best_sample = best_sample[0]*best_sample

					reference_point = low
					start_post = time.time()
					samples = extract_distinct_samples_torch(best_sample.T,single_spin=n)
					results = build_nondominated_samples_torch(results, samples,[J0.float(),J1.float(),J2.float()],False) ####### most time cost
					hv_b = get_hypervolume_torch(results, [J0.float(),J1.float(),J2.float()], reference_point)
					hv_ratio = hv_b/hv_max
					if hv_ratio>1:
						hv_ratio=1
						hv_max=hv_b
				end_ete_b=time.time()

				start_ete_d = time.time()
				H_param = 9
				weights_array = generate_simplex_lattice_weights(n_obj=num_objectives, H=H_param)
				c_vectors = weights_array
				# c_vectors = np.random.dirichlet(np.ones(num_objectives),size=100)
				J = torch.zeros((n*c_vectors.shape[0],n*c_vectors.shape[0])).to(device, non_blocking=True)
				if num_objectives==2:
					for j in range(c_vectors.shape[0]):
						J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1
				else:
					for j in range(c_vectors.shape[0]):
						J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1+c_vectors[j,2]*J2
				Ising_J = J
				xi = 1 / torch.abs(Ising_J.sum(axis=1)).max()
				Jm = Ising_J.to(device, non_blocking=True).float()
				Jm = Jm.to_sparse_csr()

				results = None
				n_iter = 50
				batch_size = 3000
				hv_ratio = 0
				while hv_ratio<0.001:
					s = tSB.SB(-Jm, n_iter=n_iter, xi=xi, dt=1, batch_size=batch_size,device=device)
					start = time.time()
					s.update_d(amp=0.15)
					end=time.time()

					rand_signs = 2.0 * torch.randint(0, 2, s.x.shape, device=s.x.device, dtype=s.x.dtype) - 1
					best_sample = torch.where(s.x == 0, rand_signs, s.x)
					best_sample = torch.sign(best_sample)
					best_sample = best_sample[0]*best_sample

					reference_point = low
					start_post = time.time()
					samples = extract_distinct_samples_torch(best_sample.T,single_spin=n)
					results = build_nondominated_samples_torch(results, samples,[J0.float(),J1.float(),J2.float()],False) ####### most time cost
					hv_d = get_hypervolume_torch(results, [J0.float(),J1.float(),J2.float()], reference_point)
					hv_ratio = hv_d/hv_max
					if hv_ratio>1:
						hv_ratio=1
						hv_max=hv_d

				end_ete_d=time.time()

				# print('\n sample time: ',end-start,
				# '\n ETE time: ', end_ete-start_ete,
				# '\n hypervolume: ', hv/hv_max,
				# '\n nondominated number: ',results.shape[0])

				n_vars = n  
				matrices = [J0.float(), J1.float(), J2.float()]
				# matrices = [matrix0, matrix1, matrix2]
				n_obj = len(matrices)

				class MyMatrixProblem(Problem):
					def __init__(self, matrices):
						self.matrices_list = matrices 
						n_vars = matrices[0].shape[0]
						n_obj = len(matrices)
						
						self.stacked_matrices = torch.stack(matrices) 
						super().__init__(n_var=n_vars, n_obj=n_obj, n_ieq_constr=0, xl=0, xu=1, vtype=int,elementwise_evaluation=False)

					def _evaluate(self, x, out, *args, **kwargs):

						# if self.stacked_matrices.device != device:
						# 	self.stacked_matrices = self.stacked_matrices.to(device)

						self.matrix_sums = self.stacked_matrices.sum(dim=[1, 2])
						
						# print(x.shape)
						x = torch.from_numpy(x).to(device, non_blocking=True).float()
						x_mapped = 2 * x - 1 # (N, D)
						
						x_expanded = x_mapped.unsqueeze(1).unsqueeze(1) # (N, 1, 1, D)

						matrices_expanded = self.stacked_matrices.unsqueeze(0) # (1, n_obj, D, D)
						xm = torch.matmul(x_expanded, matrices_expanded) 

						xm = xm.squeeze(2) 

						x_broadcast = x_mapped.unsqueeze(1) # (N, 1, D)

						quad_form = (xm * x_broadcast).sum(dim=2) 


						term = 0.5 * (self.matrix_sums - quad_form) 
						F = -term 

						out["F"] = F.detach().cpu().numpy()
				
				problem = MyMatrixProblem(matrices)
				output = None
				count = 0

				start_rvea = time.time()
				while count<0.1 :
					ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
					algorithm = RVEA(    
					ref_dirs=ref_dirs,
					sampling=BinaryRandomSampling(),          
					crossover=TwoPointCrossover(prob=0.9),    
					mutation=BitflipMutation(prob=1/n_vars)
					)
					res = minimize(problem,
								algorithm,
								termination=('n_gen', 500),
								verbose=False)

					if output is None:
						output =-res.F
					else:
						output = np.vstack((output,-res.F))
					output = output[(moocore.is_nondominated(output, maximise=False, keep_weakly=False))]
					rvea_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
					count=rvea_hv/hv_max
				end_rvea = time.time()

				problem = MyMatrixProblem(matrices)
				output = None
				count = 0

				start_moead = time.time()
				while count<0.1 :
					# ref_dirs = get_reference_directions("energy", 3, n_points=220)
					ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
					algorithm = MOEAD(
						ref_dirs,
						n_neighbors=20,
						prob_neighbor_mating=0.9,
						sampling=BinaryRandomSampling(),       
						crossover=TwoPointCrossover(prob=0.9), 
						mutation=BitflipMutation(prob=1/n_vars)
					)
	
					termination = ("n_gen", 500) 
					# print("start...")
					res = minimize(problem,
								algorithm,
								termination,
								verbose=False)

					if output is None:
						output =-res.F
					else:
						output = np.vstack((output,-res.F))
					output = output[(moocore.is_nondominated(output, maximise=False, keep_weakly=False))]
					moead_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
					count=moead_hv/hv_max
				end_moead = time.time()

				problem = MyMatrixProblem(matrices)
				output = None
				count = 0
	
				start_nsga2 = time.time()
				while count<0.1 :
				
					# ref_dirs = get_reference_directions("energy", 3, n_points=220)
					ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
	
					algorithm = NSGA2(
						pop_size=190,
						ref_dirs = ref_dirs,
						sampling=BinaryRandomSampling(),       
						crossover=TwoPointCrossover(prob=0.9), 
						mutation=BitflipMutation(prob=1/n_vars)
					)
					
					termination = ("n_gen", 500) 
					
					res = minimize(problem,
								algorithm,
								termination,
								verbose=False)
	
					if output is None:
						output =-res.F
					else:
						output = np.vstack((output,-res.F))
					output = output[(moocore.is_nondominated(output, maximise=False, keep_weakly=False))]
					nsga2_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
					count=nsga2_hv/hv_max
				end_nsga2 = time.time()
	   
				problem = MyMatrixProblem(matrices)
				output = None
				count = 0
	
				start_nsga3 = time.time()
				while count<0.1 :
				
					# ref_dirs = get_reference_directions("energy", 3, n_points=220)
					ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
	
					algorithm = NSGA3(
						popsize=190,
						ref_dirs=ref_dirs,
						sampling=BinaryRandomSampling(),      
						crossover=TwoPointCrossover(prob=0.9), 
						mutation=BitflipMutation(prob=1/n_vars)
					)
					
					termination = ("n_gen", 500) 
					
					res = minimize(problem,
								algorithm,
								termination,
								verbose=False)
	
					if output is None:
						output =-res.F
					else:
						output = np.vstack((output,-res.F))
					output = output[(moocore.is_nondominated(output, maximise=False, keep_weakly=False))]
					nsga3_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
					count=nsga3_hv/hv_max
				end_nsga3 = time.time()
	
				# print('\n moead time:',end_nsga-start_nsga,
				# 	  '\n moead results: ', count,
				# 	  '\n moead hv: ', nsga_hv )
				print(end_nsga2-start_nsga2 , nsga2_hv/hv_max)
				print(end_nsga3-start_nsga3 , nsga3_hv/hv_max)
				print(end_moead-start_moead , moead_hv/hv_max)
				print(end_rvea-start_rvea , rvea_hv/hv_max)
				print(end_ete_b-start_ete_b , hv_b/hv_max)
				print(end_ete_d-start_ete_d , hv_d/hv_max)
				n2_tlist.append(end_nsga2-start_nsga2)
				n2_hvlist.append(nsga2_hv/hv_max)
				n3_tlist.append(end_nsga3-start_nsga3)
				n3_hvlist.append(nsga3_hv/hv_max)
				m_tlist.append(end_moead-start_moead)
				m_hvlist.append(moead_hv/hv_max)
				k_tlist.append(end_rvea-start_rvea)
				k_hvlist.append(rvea_hv/hv_max)
				b_tlist.append(end_ete_b-start_ete_b)
				b_hvlist.append(hv_b/hv_max)
				d_tlist.append(end_ete_d-start_ete_d)
				d_hvlist.append(hv_d/hv_max)
				
			
	   
	tot_data = np.array([n2_tlist,n2_hvlist,n3_tlist,n3_hvlist,m_tlist,m_hvlist,k_tlist,k_hvlist,b_tlist,b_hvlist,d_tlist,d_hvlist])
	np.save('all_plot/raw_data/vsdata.npy',tot_data)

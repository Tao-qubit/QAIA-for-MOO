import numpy as np
import time
from functions import *
import torch
import itertools

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

def bf_samples(n=10):
	bool_vectors = list(itertools.product([False, True], repeat=n))
	bool_vectors = np.array(bool_vectors, dtype=bool)
	spin_vectors = 2*bool_vectors-1
	return spin_vectors


def tri_problem(n,ds):
	matrix_mask = generate_mask(n,ds)  
	A = np.random.randint(-25, 26, size=(n, n))
	B = np.random.randint(-25, 26, size=(n, n))
	### -0.9:(1,1,0.5,-2,-1,0.36) -0.5:(0.5,-1,1,1,-0.5,0.26)
	matrix0 = (1*A+1*B)
	matrix1 = (-5*A+0.2*B)
	matrix2 = np.random.randint(-25, 26, size=(n, n))
	matrix0 = (matrix0)*matrix_mask
	matrix1 = (matrix1)*matrix_mask
	matrix2 = (matrix2)*matrix_mask
	matrix0 = (matrix0+matrix0.T)
	matrix1 = (matrix1+matrix1.T)
	matrix2 = (matrix2+matrix2.T)
	return matrix0.astype(float),matrix1.astype(float),matrix2.astype(float)

# def tri_problem(n,ds):
# 	matrix_mask = generate_mask(n,ds)  
# 	matrix0 = np.random.randint(5, 51, size=(n, n))
# 	matrix1 = np.random.randint(5, 51, size=(n, n))
# 	matrix0 = (matrix0)*matrix_mask
# 	matrix0 = (matrix0+matrix0.T)
# 	matrix1 = (matrix1)*matrix_mask
# 	matrix1 = (matrix1+matrix1.T)
# 	matrix2 = .5*matrix0-5*matrix1
# 	return matrix0.astype(float),matrix1.astype(float),matrix2.astype(float)

# def pareto_count(input,ref):
# 	unique_input = np.unique(input, axis=0)
# 	combined = np.vstack((unique_input, ref))
# 	_, counts, re_index = np.unique(combined, axis=0, return_counts=True,return_inverse=True)
# 	return np.sum(counts[re_index[:unique_input.shape[0]]] > 1)

def pareto_count(input, ref):
    unique_input = np.unique(input, axis=0)
    ref_set = set(map(tuple, ref))
    count = sum(1 for row in unique_input if tuple(row) in ref_set)
    return count

import moocore

# device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n2_list=[]
n3_list=[]
md_list=[]
rv_list=[]
b_list=[]
d_list=[]

ref_pareto=[]

for (n_loop,d_loop) in [(25,0.5),(25,1),(42,'ibm'),(100,0.5),(100,1),(200,0.5),(200,1)]:
# for (n_loop,d_loop) in [(200,1)]:
		for rep_ in range(5):
			if d_loop=='ibm':
				num_objectives = 3
				n = 42
				weights = fetch_weights(num_objectives)
				Q1,Q2,Q3 = torch.zeros((42,42)).to(device),torch.zeros((42,42)).to(device),torch.zeros((42,42)).to(device)
				for key in weights:
					Q1[key]=weights[key][0]
					Q2[key]=weights[key][1]
					Q3[key]=weights[key][2]
					
				J0 , J1 , J2 = (Q1+Q1.T)/2 ,  (Q2+Q2.T)/2, (Q3+Q3.T)/2
				reference_point = get_reference_point(num_objectives)
				hv_ref = get_hv_max(num_objectives)
				sol = np.load('../3objres.npy')

			else:
				n = n_loop
				ds = d_loop
				# a ,b,c,d = 1,1,-2,-5
				np.random.seed(22233)
				matrix0, matrix1, matrix2 = tri_problem(n,ds=ds)
				np.random.seed(None)
				
				if n<30:
					spin_vectors = bf_samples(n)
					# spin_vectors = np.load(str(n)+'_'+str(ds)+'sol.npy')*2-1
				else:
					spin_vectors = np.random.randint(0, 2, size=(1000,n))*2-1

				num_objectives = 3
				J0, J1, J2 = torch.from_numpy(matrix0).to(device),  torch.from_numpy(matrix1).to(device), torch.from_numpy(matrix2).to(device)
				obj0 = 0.5*(matrix0.sum() - (spin_vectors.dot(matrix0)*spin_vectors).sum(axis=1) )
				obj1 = 0.5*(matrix1.sum() - (spin_vectors.dot(matrix1)*spin_vectors).sum(axis=1) )
				obj2 = 0.5*(matrix2.sum() - (spin_vectors.dot(matrix2)*spin_vectors).sum(axis=1) )
				cost = np.array([obj0,obj1,obj2]).T
				answer = moocore.is_nondominated(cost, maximise=True, keep_weakly=False)
				reference_point = cost.min(axis=0)
				
				hv_ref = moocore.hypervolume(-cost[answer], ref=-np.array(reference_point))
				sol = (spin_vectors[answer]/2+0.5).astype(bool)
				# if n <30:
					# np.save(str(n)+'_'+str(ds)+'sol.npy',sol)
				print('spin_non:', sol.shape)


			import sys 
			sys.path.append("..") 
			if device=='cpu':
				import tabu_SB_gpu as tSB
			else:
				import tabu_SB_gpu_half as tSB
			from itertools import combinations_with_replacement

			start_ete_b = time.time()
			H_param = 18
			weights_array = generate_simplex_lattice_weights(n_obj=num_objectives, H=H_param)
			c_vectors = weights_array
			print(c_vectors.shape)
			# c_vectors = np.random.dirichlet(np.ones(num_objectives),size=100)
			J = torch.zeros((n*c_vectors.shape[0],n*c_vectors.shape[0])).to(device)
			if num_objectives==2:
				for j in range(c_vectors.shape[0]):
					J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1
			else:
				for j in range(c_vectors.shape[0]):
					J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1+c_vectors[j,2]*J2
			Ising_J = J
			xi = 1 / torch.abs(Ising_J.sum(axis=1)).max()
			Jm = Ising_J.to(device).float()
			Jm = Jm.to_sparse_csr()

			results_b = None
			if n<50:
				n_iter = 50
				batch_size = 3000
				time_lim = 5
			else:
				n_iter = 500
				batch_size = 100
				time_lim = 10
			temp_time_b = time.time()
			while temp_time_b-start_ete_b<time_lim:
				s = tSB.SB(-Jm, n_iter=n_iter, xi=xi, dt=1, batch_size=batch_size,device=device)
				start = time.time()
				s.update_b(amp=0.1)
				end=time.time()

				rand_signs = 2.0 * torch.randint(0, 2, s.x.shape, device=s.x.device, dtype=s.x.dtype) - 1
				best_sample = torch.where(s.x == 0, rand_signs, s.x)
				best_sample = torch.sign(best_sample)
				# best_sample = best_sample[0]*best_sample

				samples = extract_distinct_samples_torch(best_sample.T,single_spin=n)
				results_b = build_nondominated_samples_torch(results_b, samples,[J0.float(),J1.float(),J2.float()],False) ####### most time cost
				temp_time_b = time.time()
				# print(temp_time_b-start_ete_b)

			hv_b = get_hypervolume_torch(results_b, [J0.float(),J1.float(),J2.float()], reference_point)
			print(results_b.shape)
			end_ete_b=time.time()

			start_ete_d = time.time()
			H_param = 18
			weights_array = generate_simplex_lattice_weights(n_obj=num_objectives, H=H_param)
			c_vectors = weights_array
			# c_vectors = np.random.dirichlet(np.ones(num_objectives),size=100)
			J = torch.zeros((n*c_vectors.shape[0],n*c_vectors.shape[0])).to(device)
			if num_objectives==2:
				for j in range(c_vectors.shape[0]):
					J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1
			else:
				for j in range(c_vectors.shape[0]):
					J[j*n:(1+j)*n,j*n:(1+j)*n]=c_vectors[j,0]*J0+c_vectors[j,1]*J1+c_vectors[j,2]*J2
			Ising_J = J
			xi = 1 / torch.abs(Ising_J.sum(axis=1)).max()
			Jm = Ising_J.to(device).float()
			Jm = Jm.to_sparse_csr()

			results_d = None
			if n<50:
				n_iter = 50
				batch_size = 3000
				time_lim = 5
			else:
				n_iter = 500
				batch_size = 100
				time_lim = 10
			temp_time_d = time.time()
			while temp_time_d-start_ete_d<time_lim:
				s = tSB.SB(-Jm, n_iter=n_iter, xi=xi, dt=1, batch_size=batch_size,device=device)
				start = time.time()
				s.update_d(amp=0.1)
				end=time.time()

				rand_signs = 2.0 * torch.randint(0, 2, s.x.shape, device=s.x.device, dtype=s.x.dtype) - 1
				best_sample = torch.where(s.x == 0, rand_signs, s.x)
				best_sample = torch.sign(best_sample)
				# best_sample = best_sample[0]*best_sample
				
				samples = extract_distinct_samples_torch(best_sample.T,single_spin=n)
				results_d = build_nondominated_samples_torch(results_d, samples,[J0.float(),J1.float(),J2.float()],False) ####### most time cost
				temp_time_d = time.time()
				# print(temp_time_d-start_ete_d)
			hv_d = get_hypervolume_torch(results_d, [J0.float(),J1.float(),J2.float()], reference_point)
			print(results_d.shape)
			end_ete_d=time.time()
   
			# bd_samples = torch.cat((results_b,results_d),dim=0)
			output_b = (results_b/2+0.5).cpu().numpy().astype(bool)
			output_b = np.logical_xor(output_b, output_b[:, [0]])
			# output_b = np.unique(output_b, axis=0)
			output_d = (results_d/2+0.5).cpu().numpy().astype(bool)
			output_d = np.logical_xor(output_d, output_d[:, [0]])
			front_ = np.vstack((output_b,output_d,sol))
			
			print('heuristics begin')
			n_vars = n  
			matrices = [J0.float(), J1.float(), J2.float()]
			# matrices = [matrix0, matrix1, matrix2]
			n_obj = len(matrices)

			class MyMatrixProblem(Problem):
				def __init__(self, matrices):
					self.matrices_list = matrices # 保存原始列表（如果需要）
					n_vars = matrices[0].shape[0]
					n_obj = len(matrices)
					
					self.stacked_matrices = torch.stack(matrices) 
					self.matrix_sums = self.stacked_matrices.sum(dim=[1, 2])
					super().__init__(n_var=n_vars, n_obj=n_obj, n_ieq_constr=0, xl=0, xu=1, vtype=int,elementwise_evaluation=False)

				def _evaluate(self, x, out, *args, **kwargs):

					x = torch.from_numpy(x).to(device).float()
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
			start_rvea = time.time()
			ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
			algorithm = RVEA(    
			ref_dirs=ref_dirs,
			popsize=190,
			sampling=BinaryRandomSampling(),         
			crossover=TwoPointCrossover(),   
			mutation=BitflipMutation(),
			eliminate_duplicates=True
			)
			# print('minimize')
			if n<50:
				n_gen = 500
			else:
				n_gen = 1000
			res = minimize(problem,
						algorithm,
						termination=('n_gen', n_gen),
						verbose=False)
			# print('minimize done')
			if output is None:
				output =-res.F
			else:
				output = np.vstack((output,-res.F))
			output_rvea = res.X[(moocore.is_nondominated(output, maximise=True, keep_weakly=False))]
			output_rvea = np.logical_xor(output_rvea, output_rvea[:, [0]])
			front_ = np.vstack((front_, output_rvea))
			rvea_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
			end_rvea = time.time()
			print('RVEA done')
   
			problem = MyMatrixProblem(matrices)
			output = None
			start_moead = time.time()
			# ref_dirs = get_reference_directions("energy", 3, n_points=220)
			ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
			algorithm = MOEAD(
				ref_dirs,
				n_neighbors=20,
				popsize=190,
				prob_neighbor_mating=0.9,
				sampling=BinaryRandomSampling(),      
				crossover=TwoPointCrossover(), 
				mutation=BitflipMutation(),
			)
			if n<50:
				time_str = '00:00:05'
			else:
				time_str = '00:00:10'		
			termination = ('time', time_str) 
			res = minimize(problem,
						algorithm,
						termination,
						verbose=False)
			if output is None:
				output =-res.F
			else:
				output = np.vstack((output,-res.F))
			output_moead = res.X[(moocore.is_nondominated(output, maximise=True, keep_weakly=False))]
			output_moead = np.logical_xor(output_moead, output_moead[:, [0]])
			front_ = np.vstack((front_, output_moead))
			moead_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
			end_moead = time.time()
			print('moead done')
   
   
			problem = MyMatrixProblem(matrices)
			output = None
			start_nsga2 = time.time()
			# ref_dirs = get_reference_directions("energy", 3, n_points=220)
			ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)

			algorithm = NSGA2(
				pop_size=190,
				ref_dirs = ref_dirs,
				sampling=BinaryRandomSampling(),       
				crossover=TwoPointCrossover(), 
				mutation=BitflipMutation(),
				eliminate_duplicates=True
			)
			if n<50:
				time_str = '00:00:05'
			else:
				time_str = '00:00:10'		
			termination = ('time', time_str) 

			res = minimize(problem,
						algorithm,
						termination,
						verbose=False)

			if output is None:
				output =-res.F
			else:
				output = np.vstack((output,-res.F))
			output_n2 = res.X[(moocore.is_nondominated(output, maximise=True, keep_weakly=False))]
			output_n2 = np.logical_xor(output_n2, output_n2[:, [0]])
			front_ = np.vstack((front_, output_n2))
			nsga2_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
			end_nsga2 = time.time()
			print('nsga2 done')
   
			problem = MyMatrixProblem(matrices)
			output = None
			start_nsga3 = time.time()
			# ref_dirs = get_reference_directions("energy", 3, n_points=220)
			ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)

			algorithm = NSGA3(
				popsize=190,
				ref_dirs=ref_dirs,
				sampling=BinaryRandomSampling(),       
				crossover=TwoPointCrossover(), 
				mutation=BitflipMutation(),
				eliminate_duplicates=True
			)
			if n<50:
				time_str = '00:00:05'
			else:
				time_str = '00:00:10'		
			termination = ('time', time_str) 

			res = minimize(problem,
						algorithm,
						termination,
						verbose=False)

			if output is None:
				output =-res.F
			else:
				output = np.vstack((output,-res.F))
			output_n3 = res.X[(moocore.is_nondominated(output, maximise=True, keep_weakly=False))]
			output_n3 = np.logical_xor(output_n3, output_n3[:, [0]])
			front_ = np.vstack((front_, output_n3))
			nsga3_hv = moocore.hypervolume(-output, ref=-np.array(reference_point))
			end_nsga3 = time.time()
			print('nsga3 done')

#### random
			output_rand = None
			rand_time_s = time.time()
			rand_time_e = time.time()
			while rand_time_e-rand_time_s<time_lim:
				random_sam = np.random.randint(0, 2, size=(190,n)).astype(bool)
				obj = [0.5*(Q.sum()-((torch.from_numpy(random_sam*2-1).to(device).float()@Q)*torch.from_numpy(random_sam*2-1).to(device).float()).sum(dim=1)).cpu().numpy() for Q in [J0.float(),J1.float(),J2.float()]]
				obj_fun = np.array(obj).T
				random_sam = random_sam[(moocore.is_nondominated(obj_fun, maximise=True, keep_weakly=False))]
				random_sam = np.logical_xor(random_sam, random_sam[:, [0]])
				if output_rand is None:
					output_rand = random_sam
				else:
					output_rand = np.vstack((output_rand,random_sam))
				rand_time_e = time.time()
			random_hv = get_hypervolume_torch(torch.from_numpy(output_rand*2-1).to(device).float(), [J0.float(),J1.float(),J2.float()],reference_point)
####		
			front_ = np.vstack((front_, output_rand))
			obj = [0.5*(Q.sum()-((torch.from_numpy(front_*2-1).to(device).float()@Q)*torch.from_numpy(front_*2-1).to(device).float()).sum(dim=1)).cpu().numpy() for Q in [J0.float(),J1.float(),J2.float()]]
			obj_fun = np.array(obj).T
			print(front_.shape)
			front_ = front_[(moocore.is_nondominated(obj_fun, maximise=True, keep_weakly=False))]
			# front_ = np.unique(front_, axis=0)
			hv_max = max(get_hypervolume_torch(torch.from_numpy(front_*2-1).to(device).float(), [J0.float(),J1.float(),J2.float()], reference_point),hv_ref)

			# print(output_b.shape,output_d.shape,front_.shape,pareto_count(output_b.T,front_)) ### shape (3,n)(3,m)(n+m,3)
			print(hv_max,front_.shape)
			print('random', random_hv/hv_max, pareto_count(output_rand,front_))
			print(end_nsga2-start_nsga2 , nsga2_hv/hv_max, pareto_count(output_n2,front_))
			print(end_nsga3-start_nsga3 , nsga3_hv/hv_max, pareto_count(output_n3,front_))
			print(end_moead-start_moead , moead_hv/hv_max, pareto_count(output_moead,front_))
			print(end_rvea-start_rvea , rvea_hv/hv_max, pareto_count(output_rvea,front_))
			print(end_ete_b-start_ete_b , hv_b/hv_max, pareto_count(output_b,front_))
			print(end_ete_d-start_ete_d , hv_d/hv_max, pareto_count(output_d,front_))
			n2_list.append([end_nsga2-start_nsga2 , nsga2_hv/hv_max, pareto_count(output_n2,front_)])
			n3_list.append([end_nsga3-start_nsga3 , nsga3_hv/hv_max, pareto_count(output_n3,front_)])
			md_list.append([end_moead-start_moead , moead_hv/hv_max, pareto_count(output_moead,front_)])
			rv_list.append([end_rvea-start_rvea , rvea_hv/hv_max, pareto_count(output_rvea,front_)])
			b_list.append([end_ete_b-start_ete_b , hv_b/hv_max, pareto_count(output_b,front_)])
			d_list.append([end_ete_d-start_ete_d , hv_d/hv_max, pareto_count(output_d,front_)])
			ref_pareto.append([hv_max,front_.shape[0],random_hv/hv_max, pareto_count(output_rand,front_)])

			
tot_data = np.array([n2_list,n3_list,md_list,rv_list,b_list,d_list])
ref_pareto = np.array(ref_pareto)
np.save('vsdata.npy',tot_data)
np.save('refdata.npy',ref_pareto)
import numpy as np
import torch 

import time
import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")

from utils.functions import *

if torch.cuda.is_available():
    import utils.NI_SB_gpu_half as tSB
else:
    import utils.NI_SB_gpu as tSB


if __name__=='__main__':

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	num_objectives = 3
	weights = fetch_weights(num_objectives)
	H_param = 18

	print(f"Using device: {device}, num_objectives={num_objectives}, H_param={H_param}")

	Q1,Q2,Q3,Q4 = torch.zeros((42,42)).to(device, non_blocking=True),torch.zeros((42,42)).to(device, non_blocking=True),\
				torch.zeros((42,42)).to(device, non_blocking=True),torch.zeros((42,42)).to(device, non_blocking=True)
	for key in weights:
		Q1[key]=weights[key][0]
		Q2[key]=weights[key][1]
		Q3[key]=weights[key][2]
		Q4[key]=weights[key][-1]
		
	J1 , J2 , J3,J4 = (Q1+Q1.T)/2 ,  (Q2+Q2.T)/2, (Q3+Q3.T)/2,(Q4+Q4.T)/2

	for bd in ['b','d']:
		for i in [1,2,3,4,5]:
			print('num= ',i)
			sampling_time=[0]
			nd_list = [0]
			hv_list = [0]
			s_pre = time.time()
			# c_vectors = make_c_vectors(num=i, num_objectives=num_objectives)
			weights_array = generate_simplex_lattice_weights(n_obj=num_objectives, H=H_param)
			c_vectors = weights_array
			# c_vectors = np.random.dirichlet(np.ones(num_objectives),size=i)
			J = torch.zeros((42*c_vectors.shape[0],42*c_vectors.shape[0])).to(device, non_blocking=True)
			if num_objectives==4:
				for j in range(c_vectors.shape[0]):
					J[0+42*j:(1+j)*42,0+42*j:(1+j)*42]=c_vectors[j,0]*J1+c_vectors[j,1]*J2+c_vectors[j,2]*J3+c_vectors[j,3]*J4
			else:
				for j in range(c_vectors.shape[0]):
					J[0+42*j:(1+j)*42,0+42*j:(1+j)*42]=c_vectors[j,0]*J1+c_vectors[j,1]*J2+c_vectors[j,2]*J3
			# J = make_couplings_re(c_vectors)
			# Ising_J = dict_to_adj_matrix_torch(J)
			Ising_J = J
			xi = 1 / torch.abs(Ising_J.sum(axis=1)).max()
			Jm = Ising_J.to(device, non_blocking=True).float()
			Jm = Jm.to_sparse_csr()
			e_pre = time.time()
			endTOend_time=[e_pre-s_pre]
			for k in range(3000):
				s_ete = time.time()
				n_iter=50
				batch_size=3000
				if k==0:
					results = None
					n_iter=30
					batch_size=1000
					# tabu = None
				s = tSB.SB(-Jm, n_iter=n_iter, xi=xi, dt=1, batch_size=batch_size,device=device)
				if bd=='b':
					start = time.time()
					s.update_b(amp=0.15)
					end=time.time()
				if bd=='d':
					start = time.time()
					s.update_d(amp=0.15)
					end=time.time()
				rand_signs = 2.0 * torch.randint(0, 2, s.x.shape, device=s.x.device, dtype=s.x.dtype) - 1
				best_sample = torch.where(s.x == 0, rand_signs, s.x)
				best_sample = torch.sign(best_sample)
				best_sample = best_sample[0]*best_sample
				hv_max = get_hv_max(num_objectives)
				reference_point = get_reference_point(num_objectives)
		
				samples = extract_distinct_samples_torch(best_sample.T,single_spin=42)
				if num_objectives==4:
					results = build_nondominated_samples_torch(results, samples, [J1,J2,J3,J4] ,False) ####### most time cost
				else:
					results = build_nondominated_samples_torch(results, samples, [J1,J2,J3] ,False) ####### most time cost
				e_ete=time.time()
				if num_objectives==4:
					hv = get_hypervolume_torch(results, [J1,J2,J3,J4], reference_point)
				else:
					hv = get_hypervolume_torch(results, [J1,J2,J3], reference_point)

				print('\n sample time: ',end-start,
				'\n ETE time: ', e_ete-s_ete ,
				'\n hypervolume diff: ', hv_max-hv,
				'\n nondominated number: ',results.shape[0])
				sampling_time.append(end-start+sampling_time[-1])
				endTOend_time.append(e_ete-s_ete+endTOend_time[-1])
				nd_list.append(results.shape[0])
				hv_list.append(hv_max-hv)

				if np.isclose(hv_max, hv, rtol=1e-14, atol=1e-06):
					break

			records = np.array([sampling_time,endTOend_time,nd_list,hv_list])
			np.save(f'all_plot/raw_data/{num_objectives}'+bd+'obj_trail_'+str(i)+'.npy', records)
			print(records)

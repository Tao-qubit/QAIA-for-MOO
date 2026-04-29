import numpy as np
import torch 
import urllib.request
import json
import moocore

# import time
from itertools import combinations_with_replacement
# import itertools


def fetch_weights(Nobj):
    filename = {3: {}, 4: {}}
    filename[3, 0] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_3o_0/problem_graph_0.json')
    filename[3, 1] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_3o_0/problem_graph_1.json')
    filename[3, 2] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_3o_0/problem_graph_2.json')
    filename[4, 0] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_0.json')
    filename[4, 1] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_1.json')
    filename[4, 2] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_2.json')
    filename[4, 3] = ('https://raw.githubusercontent.com/stefan-woerner/qamoo/8bddff3a7d49c18ed3c3d0bf0494e35965117131/'
                      'data/problems/42q/problem_set_42q_0s_4o_0/problem_graph_3.json')

    weights = []
    for obj in range(Nobj):
        weights.append([])
        with urllib.request.urlopen(filename[Nobj, obj]) as response:
            data = json.loads(response.read().decode())
        weights[obj] = {
            tuple(sorted([edge['source'], edge['target']])): edge['weight'] for edge in data['links']
        }

    meta_weights = {
        key: np.array([w[key] for w in weights]) for key in weights[0]
    }
    return meta_weights

def make_c_vectors(experiment_repetition=1, iteration=100, num=96, num_objectives=None):
	# np.random.seed(10000 * experiment_repetition + iteration)
	# np.random.seed(1226)
	rand_points = np.random.rand(num, num_objectives + 1)
	rand_points[:, [0, 1]] = [0, 1]
	rand_points[:, 1] = 1
	return np.diff(np.sort(rand_points, axis=1), axis=1)

def make_couplings(c: np.array, embeddings):
	weights = fetch_weights(len(c[0]))
	J = {}

	for iemb, embedding in enumerate(embeddings):
		for u, v in weights:
			J[embedding[u], embedding[v]] = np.dot(weights[u, v], c[iemb])
	return J

def make_couplings_re(c: np.array):
	weights = fetch_weights(len(c[0]))
	triv_emb = np.arange(42)
	size = len(c)
	J = {}
	for iemb in range(size):
		for u, v in weights:
			J[triv_emb[u]+42*iemb, triv_emb[v]+42*iemb] = np.dot(weights[u, v], c[iemb])
	return J

def dict_to_adj_matrix_torch(edge_dict, num_nodes=None, dtype=torch.float32):
	if num_nodes is None:
		all_nodes = set()
		for i, j in edge_dict.keys():
			all_nodes.add(i)
			all_nodes.add(j)
		num_nodes = max(all_nodes) + 1 if all_nodes else 0

	adj = torch.zeros((num_nodes, num_nodes), dtype=dtype)
	for (i, j), w in edge_dict.items():
		adj[i, j] = w
		adj[j, i] = w  
	return adj

def build_nondominated_samples(previous_samples, new_samples, _weights, index=False):
	if previous_samples is None:
		all_samples = np.unique(new_samples, axis=0)
	else:
		all_samples = np.unique(np.vstack((previous_samples, new_samples)), axis=0)
	cut_edges = all_samples[:, np.array(list(_weights.keys()))[:, 0]] != all_samples[:,
																		 np.array(list(_weights.keys()))[:, 1]]
	obj_fun = np.matmul(cut_edges, np.array(list(_weights.values())))
	if index==False:
		return all_samples[moocore.is_nondominated(obj_fun, maximise=True, keep_weakly=False)]
	else:
		return moocore.is_nondominated(obj_fun, maximise=True, keep_weakly=False)

def extract_distinct_samples(ss, embs):
	spin_array = ss.record.sample
	variables = ss.variables
	all_spins_array = np.ones((spin_array.shape[0], np.max(embs) + 1))
	all_spins_array[:, variables] = spin_array

	samples = np.unique(np.reshape(np.array([all_spins_array[:, emb].copy() for emb in embs]) == 1, (-1, len(embs[0]))),
						axis=0)
	return np.unique(np.logical_xor(samples, np.tile(samples[:, 0], (samples.shape[1], 1)).T), axis=0)

def extract_distinct_samples_index(bool_spins):
	samples, unique_idx_to_original = np.unique(bool_spins, axis=0, return_index=True)
	_, unique_idx_in_samples = np.unique(np.logical_xor(samples, samples[:, [0]]), axis=0, return_index=True)
	return unique_idx_to_original[unique_idx_in_samples]

def extract_distinct_samples_re(sb_samp,single_spin=42):
	all_spins_array = sb_samp
	samples = np.unique(np.reshape(all_spins_array == 1, (-1, single_spin)),
						axis=0)
	return np.unique(np.logical_xor(samples, samples[:, [0]]), axis=0)

def get_hypervolume(_samples, _weights, _reference_point):
	cut_edges = _samples[:, np.array(list(_weights.keys()))[:, 0]] != _samples[:, np.array(list(_weights.keys()))[:, 1]]
	obj_fun = np.matmul(cut_edges, np.array(list(_weights.values())))
	return moocore.hypervolume(-obj_fun, ref=-np.array(_reference_point))

def get_reference_point(Nobj):
	return {
		3: [-12.137398079531431, -19.64152167587139, -18.33061914071653],
		4: [-17.34831473307451, -25.11279714770653, -18.471718787635094, -17.89300836655866]
	}[Nobj]

def get_hv_max(Nobj):
	return {3: 43471.70365440166, 4: 1266143.349404145}[Nobj]


def extract_distinct_samples_torch(sb_samp, single_spin=42):

    reshaped = sb_samp.reshape(-1, single_spin)
    # mask = (reshaped == 1)
    unique_samples = torch.unique(reshaped, dim=0)
    # first_col = unique_samples[:, [0]].expand_as(unique_samples)
    # standardized = torch.logical_xor(unique_samples, first_col)
    # final_result = torch.unique(standardized, dim=0)

    return unique_samples

def build_nondominated_samples_torch(previous_samples, new_samples, _weights, index=False):
	if previous_samples is None:
		all_samples = torch.unique(new_samples, dim=0)
	else:
		all_samples = torch.unique(torch.cat((previous_samples, new_samples), dim=0), dim=0)

	cost = [0.5*(Q.sum()-((all_samples.float()@Q)*all_samples.float()).sum(dim=1)).cpu().numpy() for Q in _weights]
	obj_fun = np.array(cost).T
 
	if index==False:
		return all_samples[moocore.is_nondominated(obj_fun, maximise=True, keep_weakly=False)]
	else:
		return moocore.is_nondominated(obj_fun, maximise=True, keep_weakly=False)

def get_hypervolume_torch(_samples, _weights, _reference_point):
	cost = [0.5* ( (Q.sum()-((_samples.float()@Q)*_samples.float()).sum(dim=1))).cpu().numpy() for Q in _weights]
	obj_fun = np.array(cost).T
	return moocore.hypervolume(-obj_fun, ref=-np.array(_reference_point))



def generate_simplex_lattice_weights(n_obj=3, H=12):
    
    weights_list = []
    
    for comb in combinations_with_replacement(range(H + 1), n_obj - 1):

        boundaries = [0] + list(comb) + [H]
        

        n_values = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        

        w_vector = np.array(n_values) / H
        weights_list.append(w_vector)
    
    return np.array(weights_list)



def generate_mask(n, density, seed=None):
	if seed is not None:
		np.random.seed(seed=seed)
	max_edges = n * (n - 1) // 2
	m = int(round(density * max_edges))
	m = max(0, min(m, max_edges)) 
	adj = np.zeros((n, n), dtype=int)
	
	i_upper, j_upper = np.triu_indices(n, k=1)  
	total_possible = len(i_upper)
	
	selected = np.random.choice(total_possible, size=m, replace=False)
	
	for idx in selected:
		i, j = i_upper[idx], j_upper[idx]
		adj[i, j] = 1
		# adj[j, i] = 1  
	
	return adj

def generate_C2_from_C1(C1, r=46):
	n = C1.shape[0]
	C2 = np.zeros((n, n), dtype=int)

	for i in range(n):
		for j in range(n):
			c1 = C1[i, j]
			if c1 < (r - 1) / 2:
				low = r - 1 - c1
				high = r - 1
				C2[i, j] = np.random.randint(low, high + 1)
			else:

				high = r - 1 - c1
				# if high<0:
				# 	C2[i,j] = 0
				# else:
				C2[i, j] = np.random.randint(0, high + 1)
				
	return C2
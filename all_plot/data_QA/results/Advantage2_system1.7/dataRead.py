import numpy as np 

### ['qpu_access_time', 'qpu_annealing_time', 'hypervolume', 'nondominated_samples']
data = np.load('Nobj4_rep3.npz')
print(data['nondominated_samples'].shape)
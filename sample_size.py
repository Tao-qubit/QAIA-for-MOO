import numpy as np
import torch
import pandas as pd

import utils.NI_SB_gpu as SB
from utils.functions import fetch_weights, make_couplings_re, get_hv_max, get_reference_point,\
                         dict_to_adj_matrix_torch, generate_simplex_lattice_weights, \
                            extract_distinct_samples_re, build_nondominated_samples, get_hypervolume


if __name__=="__main__":
    num_objectives = 3 # Number of objective functions, either 3 or 4

    weights = fetch_weights(num_objectives)
    hv_max = get_hv_max(num_objectives)
    reference_point = get_reference_point(num_objectives)
    print(hv_max)

    batch_size = 500
    h_value = 16

    weights_array = generate_simplex_lattice_weights(n_obj=num_objectives, H=h_value)
    c_vectors = weights_array
    num_c_vectors = len(c_vectors)
    print(num_c_vectors)

    J = make_couplings_re(c_vectors)
    Ising_J = dict_to_adj_matrix_torch(J)
    xi = 1 / torch.abs(Ising_J.sum(axis=1)).max()
    Jm = Ising_J.cpu().float()
    Jm = Jm.to_sparse_csr()

    sb_types = ['bSB', 'dSB']

    num_trials = 100
    found_nondom = {sb_type: [[] for trial_ind in range(num_trials)] for sb_type in sb_types}
    iterations = {sb_type: [[] for trial_ind in range(num_trials)] for sb_type in sb_types}
    max_k = 0

    amp=0.15
    n_iter = 50
    max_pareto_vectors_dict = {3: 2067, 4: 30419}
    QA_samples_dict = {3: int(0.288*10**6), 4: int(20*10**6)}
    QAOA_samples_dict = {3: 25*10**6, 4: 100*10**6}
    QA_samples = QA_samples_dict[num_objectives]
    QAOA_samples = QAOA_samples_dict[num_objectives]
    max_pareto_vectors = max_pareto_vectors_dict[num_objectives]

    settings_df = pd.DataFrame({'num_trials': num_trials, 'amp': amp, 'num_c_vectors': num_c_vectors, 'num_objectives': num_objectives, 'h_value': h_value, 'batch_size': batch_size, 'n_iter': n_iter, \
                            'hv_max': hv_max, 'max_Pareto_solutions': max_pareto_vectors, 'QA_samples': QA_samples, 'QAOA_samples': QAOA_samples}, index=[0])
    
    raw_data = []
    for sb_type in sb_types:	
        for trial_ind in range(num_trials):
            all_nondominant_solutions = None
            for k in range(2000):

                s = SB.SB(-Jm, n_iter=n_iter, xi=xi, dt=1, batch_size=batch_size, device='cpu')
                if sb_type == 'bSB':
                    s.update_b(amp=amp)
                elif sb_type == 'dSB':
                    s.update_d(amp=amp)
                else:
                    print('SB type not supported ', sb_type)
                    break
                best_sample = torch.sign(s.x).cpu()

                samples = extract_distinct_samples_re(best_sample.T)
                all_nondominant_solutions = build_nondominated_samples(all_nondominant_solutions, samples, weights)
                hv = get_hypervolume(all_nondominant_solutions, weights, reference_point)
                found_nondom[sb_type][trial_ind].append(len(all_nondominant_solutions))
                iterations[sb_type][trial_ind].append(k)
                max_k = max(k, max_k)

                if k%50 == 49:
                    print(f'{k}, sampled_so_far={num_c_vectors*batch_size*(k+1)}, found_so_far={all_nondominant_solutions.shape[0]}, hv={hv}, hv_diff={hv_max-hv}')

                if np.abs(hv_max - hv) < 0.0000001:
                    print(f'{sb_type}, {trial_ind}, with {num_c_vectors} fixed c_vec, need {num_c_vectors*batch_size*(k+1)} samples to get hv_max in {k} iters, finding {all_nondominant_solutions.shape[0]} nondominated solutions')
                    samples_result = {'trial': trial_ind, 'SB': sb_type, 'sample_size': num_c_vectors*batch_size*(k+1)}
                    raw_data.append(samples_result)
                    break
    samples_df = pd.DataFrame(raw_data)

    summary = samples_df.groupby('SB')['sample_size'].agg(
                    min_obj='min',
                    median_obj='median',
                    mean_obj='mean',
                    max_obj='max'
                ).reset_index()
    
    samples_df.to_csv('all_plot/raw_data/Samples_data.csv', index=False)
    settings_df.to_csv('all_plot/raw_data/Settings_data.csv', index=False)
    

    # from matplotlib import pyplot as plt
    # from matplotlib.gridspec import GridSpec
    # from matplotlib.colors import Normalize
    # import seaborn as sns

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # sns.violinplot(data=samples_df, ax=ax, x='SB', y='sample_size', 
    #             # palette='viridis',  # Color palette
    #             inner=None,         # Show quartiles inside
    #             linewidth=1.5)       # Line width)
    # left_l = -0.5
    # right_l = 1.5
    # ax.plot([left_l, right_l], [QA_samples, QA_samples], color='green', linewidth=2, label=f'QA, {QA_samples}')
    # ax.plot([left_l, right_l], [QAOA_samples, QAOA_samples], color='orange', linewidth=2, label=f'QAOA, {QAOA_samples}')

    # means = samples_df.groupby('SB')['sample_size'].mean()
    # x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    # mean_values = [means[algo] for algo in x_labels]
    # x_positions = range(len(x_labels))
    # ax.hlines(
    #     y=mean_values, 
    #     xmin=[i - 0.3 for i in x_positions], 
    #     xmax=[i + 0.3 for i in x_positions], 
    #     colors='blue', 
    #     linewidths=2.5, 
    #     linestyles='dashed', 
    #     zorder=3,  # Ensures lines are drawn on top of violins
    #     label='Mean'
    # )


    # ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.60), 
    #         bbox_transform=ax.transAxes, frameon=True)
    # ax.set_ylabel('Total_samples')
    # ax.set_yscale('log')

    # fig.tight_layout()
    # fig.savefig('Sample_size_two_SB.png', dpi=300)
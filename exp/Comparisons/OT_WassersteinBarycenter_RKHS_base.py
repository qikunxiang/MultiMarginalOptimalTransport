import numpy as np
import time
from RKHS_MMOT import *
from OT_WassersteinBarycenter_marginals import *
from OT_WassersteinBarycenter_cost import *
import json

NUM_OF_MARGS = 5
DIM = 2
PENALTY_TYPE = 'L2'
PENALTY_PARAM_GAMMA = 25000
DECAY_SCHEME = 'exp'
KERNEL_TYPE = 'Laplace'
KERNEL_SIG = 8.0

NUM_OF_ITERS = 600000
LEARNING_RATE_BEGIN = 0.00006
LEARNING_RATE_FINAL = 10**(-7)
DECAY_UPDATE_FREQUENCY = 1000
DECAY_POLY_EXPONENT = 2
NUM_OF_MC_SAMP_FINAL_OBJ = 50000

dual_funcs_eval_points_x = np.linspace(0, 3, 201)[..., np.newaxis]
dual_funcs_eval_points_y = np.linspace(0, 3, 201)[..., np.newaxis]
dual_funcs_eval_points = np.hstack((np.repeat(dual_funcs_eval_points_x, 201, axis=0), np.tile(dual_funcs_eval_points_y, (201, 1))))

def sample_generator():
    with open('WBMM_densities.json', 'r') as jsonfile:
        WBMM_densities = json.load(jsonfile)

    grid_densities_list = []

    for marg_i in range(NUM_OF_MARGS):
        marg_densities = np.reshape(np.array(WBMM_densities[marg_i]), shape=[4,-1])
        grid_densities_list.append(marg_densities)
    
    return meas_CPWA_generator(rng, 1, grid_densities_list)

def RKHS_kernel(x, y):
    if KERNEL_TYPE == 'Laplace':
        return RKHS_Laplace_kernel_func(x, y, sig=KERNEL_SIG)
    elif KERNEL_TYPE == 'Gauss':
        return RKHS_Gauss_kernel_func(x, y, sig=KERNEL_SIG)

num_of_runs = 20
save_list = [PENALTY_TYPE, PENALTY_PARAM_GAMMA, KERNEL_TYPE, KERNEL_SIG, DECAY_SCHEME]
save_list_string = [str(x) for x in save_list]

filename_prefix = 'Saved/WassersteinBarycenter/RKHS/' + '_'.join(save_list_string)

filename_objectives = filename_prefix + '_objectives.txt'
filename_objectives_unpenalized = filename_prefix + '_objectives_unpenalized.txt'
filename_objectives_penalties = filename_prefix + '_objectives_penalties.txt'
filename_running_time = filename_prefix + '_running_time.txt'
filename_json = filename_prefix + '.json'


objective_vals_list = []
objective_unpenalized_vals_list = []
objective_penalties_list = []
optimizer_list = []
time_spent_list = []

dual_funcs_vals = None

for run_i in range(num_of_runs):
    t0 = time.time()
    rng = np.random.RandomState(990000 + 10 * run_i)
    value, value_unpen, penalty, optimizer, dual_funcs_vals_output = RKHS_MMOT(sample_generator, OT_WassersteinBarycenter_cost_np, NUM_OF_MARGS, DIM, RKHS_kernel, penalty_param_gamma=PENALTY_PARAM_GAMMA, num_of_iters=NUM_OF_ITERS, learning_rate_begin=LEARNING_RATE_BEGIN, learning_rate_final=LEARNING_RATE_FINAL, decay_scheme=DECAY_SCHEME, decay_update_frequency=DECAY_UPDATE_FREQUENCY, decay_poly_exponent=DECAY_POLY_EXPONENT, penalty_type=PENALTY_TYPE, num_of_MC_samp_final_obj=NUM_OF_MC_SAMP_FINAL_OBJ, dual_funcs_eval_points=([dual_funcs_eval_points for _ in range(NUM_OF_MARGS)] if run_i == 0 else None), log_tag='run ' + str(run_i))

    time_spent_list.append(time.time()-t0)
    objective_vals_list.append(value)
    objective_unpenalized_vals_list.append(value_unpen)
    objective_penalties_list.append(penalty)
    optimizer_list.append(optimizer)

    if dual_funcs_vals_output is not None:
        dual_funcs_vals = dual_funcs_vals_output

    output_dict = {
        "objective_vals_list": objective_vals_list,
        "objective_unpenalized_vals_list": objective_unpenalized_vals_list,
        "objective_penalties_list": objective_penalties_list,
        "time_spent_list": time_spent_list,
        "dual_funcs_eval_points": np.squeeze(dual_funcs_eval_points).tolist(),
        "dual_funcs_vals": dual_funcs_vals
    }

    print('\n')
    print(objective_vals_list)
    print(objective_unpenalized_vals_list)
    print(objective_penalties_list)
    print(time_spent_list)
    print('\n')
    np.savetxt(filename_objectives, objective_vals_list)
    np.savetxt(filename_objectives_unpenalized, objective_unpenalized_vals_list)
    np.savetxt(filename_objectives_penalties, objective_penalties_list)
    np.savetxt(filename_running_time, time_spent_list)

    with open(filename_json, 'w') as json_file:
        json.dump(output_dict, json_file)
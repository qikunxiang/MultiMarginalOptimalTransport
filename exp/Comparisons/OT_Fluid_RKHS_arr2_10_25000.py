import numpy as np
import time
import random
from RKHS_MMOT import *
from OT_Fluid_cost import *
import json

ARRANGEMENT = 2
NUM_OF_MARGS = 10
DIM = 1
PENALTY_TYPE = 'L2'
PENALTY_PARAM_GAMMA = 25000
DECAY_SCHEME = 'exp'
KERNEL_TYPE = 'Laplace'
KERNEL_SIG = 0.5

NUM_OF_ITERS = 600000
LEARNING_RATE_BEGIN = 0.00006
LEARNING_RATE_FINAL = 10**(-7)
DECAY_UPDATE_FREQUENCY = 1000
DECAY_POLY_EXPONENT = 2
NUM_OF_MC_SAMP_FINAL_OBJ = 50000

dual_funcs_eval_points = np.linspace(0, 1, 401)[..., np.newaxis]

def Uniform_sample_generator():
    while 1:
        yield np.random.rand(NUM_OF_MARGS, DIM)

def cost_func(x):
    if ARRANGEMENT == 1:
        return OT_Fluid_cost_np(x[np.newaxis, ...], OT_Fluid_arrangement1_np)
    elif ARRANGEMENT == 2:
        return OT_Fluid_cost_np(x[np.newaxis, ...], OT_Fluid_arrangement2_np)

def RKHS_kernel(x, y):
    if KERNEL_TYPE == 'Laplace':
        return RKHS_Laplace_kernel_func(x, y, sig=KERNEL_SIG)
    elif KERNEL_TYPE == 'Gauss':
        return RKHS_Gauss_kernel_func(x, y, sig=KERNEL_SIG)

num_of_runs = 20
save_list = [NUM_OF_MARGS, PENALTY_TYPE, PENALTY_PARAM_GAMMA, KERNEL_TYPE, KERNEL_SIG, DECAY_SCHEME]
save_list_string = [str(x) for x in save_list]

filename_prefix = 'Saved/Fluid/RKHS/arr' + str(ARRANGEMENT) + '/' + '_'.join(save_list_string)

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
    np.random.seed(80000 + 10000 * ARRANGEMENT + 100 * NUM_OF_MARGS + 10 * run_i)
    t0 = time.time()
    value, value_unpen, penalty, optimizer, dual_funcs_vals_output = RKHS_MMOT(Uniform_sample_generator, cost_func, NUM_OF_MARGS, DIM, RKHS_kernel, penalty_param_gamma=PENALTY_PARAM_GAMMA, num_of_iters=NUM_OF_ITERS, learning_rate_begin=LEARNING_RATE_BEGIN, learning_rate_final=LEARNING_RATE_FINAL, decay_scheme=DECAY_SCHEME, decay_update_frequency=DECAY_UPDATE_FREQUENCY, decay_poly_exponent=DECAY_POLY_EXPONENT, penalty_type=PENALTY_TYPE, num_of_MC_samp_final_obj=NUM_OF_MC_SAMP_FINAL_OBJ, dual_funcs_eval_points=([dual_funcs_eval_points for _ in range(NUM_OF_MARGS)] if run_i == 0 else None), log_tag=DECAY_SCHEME + ' run ' + str(run_i))

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
import tensorflow.compat.v1 as tf # type: ignore
import numpy as np
import time
import json
from OT_WassersteinBarycenter_marginals import *
from OT_WassersteinBarycenter_cost import *

# Since the original code was written in TensorFlow v1, compatibility mode is turned on
tf.disable_v2_behavior()

NUM_OF_MARGS = 5
DIM = 2
PENALTY_TYPE = 'L2'  # Either 'L2' or 'exp'
PENALTY_PARAM_GAMMA = 500000

BATCH_REFJOINT = 4096
BATCH_MARGS = 2048
NUM_OF_ITERS = 50000 + 10000 * DIM * NUM_OF_MARGS # original setting: min(50000 + 10000 * DIM * NUM_OF_MARGS, 95000)

ACTIVATION = 'ReLu'
NUM_OF_LAYERS = 5
NUM_OF_NEURONS_PER_LAYER = 64 * DIM

DECAY_START = 5000
DECAY_STEPS = NUM_OF_ITERS - DECAY_START
LEARNING_RATE_BEGIN = 0.0001
LEARNING_RATE_FINAL = 0.0000005
DECAY_EXPONENT = 2
NUM_OF_MC_SAMP_FINAL_OBJ = 50000
COMPUTE_PRIMAL = True

with open('WBMM_densities.json', 'r') as jsonfile:
    WBMM_densities = json.load(jsonfile)

grid_densities_list = []

for marg_i in range(NUM_OF_MARGS):
    marg_densities = np.reshape(np.array(WBMM_densities[marg_i]), newshape=[4,-1])
    grid_densities_list.append(marg_densities)

dual_funcs_eval_points_x = np.linspace(0, 3, 201)[..., np.newaxis]
dual_funcs_eval_points_y = np.linspace(0, 3, 201)[..., np.newaxis]
dual_funcs_eval_points = np.hstack((np.repeat(dual_funcs_eval_points_x, 201, axis=0), np.tile(dual_funcs_eval_points_y, (201, 1))))


print('DIM = ' + str(DIM))
print('NUM_OF_MARGS = ' + str(NUM_OF_MARGS))
print('PENALTY_TYPE = ' + str(PENALTY_TYPE))
print('PENALTY_PARAM_GAMMA = ' + str(PENALTY_PARAM_GAMMA))
print('BATCH_REFJOINT = ' + str(BATCH_REFJOINT))
print('BATCH_MARGS = ' + str(BATCH_MARGS))
print('NUM_OF_NEURONS_PER_LAYER = ' + str(NUM_OF_NEURONS_PER_LAYER))
print('NUM_OF_LAYERS: ' + str(NUM_OF_LAYERS))
print('ACTIVATION: ' + ACTIVATION)

# Build tf graph
def costfunc_tf(x):
    return -OT_WassersteinBarycenter_cost_tf(x)


def costfunc_np(x):
    return -OT_WassersteinBarycenter_cost_np(x)


# Build a single layer of the neural network
def NN_layer(x, layer_idx, input_dim, output_dim, activation):
    ua_w = tf.get_variable('ua_w'+str(layer_idx), shape=[input_dim, output_dim], initializer=tf.keras.initializers.glorot_normal(seed=0), dtype=tf.float64)
    ua_b = tf.get_variable('ua_b'+str(layer_idx), shape=[output_dim], initializer=tf.keras.initializers.glorot_normal(seed=0), dtype=tf.float64)
    z = tf.matmul(x, ua_w) + ua_b
    if activation == 'ReLu':
        return tf.nn.relu(z)
    if activation == 'tanh':
        return tf.nn.tanh(z)
    if activation == 'leakyReLu':
        return tf.nn.leaky_relu(z)
    if activation == 'softplus':
        return tf.nn.softplus(z)
    if activation == 'sin':
        return tf.sin(z)
    else:
        return z

# Build the entire neural network
def NN_approx(x, name, num_of_layers, hidden_dim, input_dim, output_dim, activation):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        layer = NN_layer(x, layer_idx=0, input_dim=input_dim, output_dim=hidden_dim, activation=activation)
        for layer_i in range(1, num_of_layers-1):
            layer = NN_layer(layer, layer_idx=layer_i, input_dim=hidden_dim, output_dim=hidden_dim, activation=activation)
        layer = NN_layer(layer, layer_idx=num_of_layers-1, input_dim=hidden_dim, output_dim=output_dim, activation='')
        return layer

# symb_samples_from_margs will contain samples from the marginals in order to approximate the integrals of the neural networks with respect to the marginal measures via Monte Carlo integration
symb_samples_from_margs = tf.placeholder(dtype=tf.float64, shape=[None, NUM_OF_MARGS, DIM])

# symb_samples_from_refjoint will contain samples from the reference joint measure (i.e., the product measure of the marginals) in order to approximate the integral of the penalty function with respect to the reference joint measure
symb_samples_from_refjoint = tf.placeholder(dtype=tf.float64, shape=[None, NUM_OF_MARGS, DIM])

potential_marg_samples = []
sum_of_potentials_marg_samples = 0
for marg_i in range(NUM_OF_MARGS):
    nn = tf.reduce_sum(NN_approx(symb_samples_from_margs[:, marg_i, :], name=str(marg_i), num_of_layers=NUM_OF_LAYERS, hidden_dim=NUM_OF_NEURONS_PER_LAYER, input_dim=DIM, output_dim=1, activation=ACTIVATION), axis=1)
    potential_marg_samples.append(nn)
    sum_of_potentials_marg_samples += nn
sum_of_integrals_MonteCarlo = tf.reduce_mean(sum_of_potentials_marg_samples)

sum_of_potentials_refjoint_samples = 0
for marg_i in range(NUM_OF_MARGS):
    sum_of_potentials_refjoint_samples += tf.reduce_sum(NN_approx(symb_samples_from_refjoint[:, marg_i, :], name=str(marg_i), num_of_layers=NUM_OF_LAYERS, hidden_dim=NUM_OF_NEURONS_PER_LAYER, input_dim=DIM, output_dim=1, activation=ACTIVATION), axis=1)


costfunc_vals = costfunc_tf(symb_samples_from_refjoint)
superreplic_diff_samples = costfunc_vals - sum_of_potentials_refjoint_samples
if PENALTY_TYPE == 'L2':
    penalty_MonteCarlo = PENALTY_PARAM_GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(superreplic_diff_samples)))
elif PENALTY_TYPE == 'exp':
    penalty_MonteCarlo = 1./PENALTY_PARAM_GAMMA * tf.reduce_mean(tf.exp(PENALTY_PARAM_GAMMA * superreplic_diff_samples - 1))
else:
    print('Reassigned PEN to L2 as current version is not implemented')
    PENALTY_TYPE = 'L2'
    penalty_MonteCarlo = PENALTY_PARAM_GAMMA * tf.reduce_mean(tf.square(tf.nn.relu(superreplic_diff_samples)))

objective_func = sum_of_integrals_MonteCarlo + penalty_MonteCarlo

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.polynomial_decay(LEARNING_RATE_BEGIN, global_step, DECAY_STEPS, LEARNING_RATE_FINAL, power=DECAY_EXPONENT)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.99, beta2=0.995).minimize(objective_func)

num_of_runs = 20
save_list = [PENALTY_TYPE, PENALTY_PARAM_GAMMA, ACTIVATION, NUM_OF_LAYERS, NUM_OF_NEURONS_PER_LAYER]
save_list_string = [str(x) for x in save_list]

filename_prefix = 'Saved/WassersteinBarycenter/NN/' + '_'.join(save_list_string)

filename_objectives = filename_prefix + '_objectives.txt'
filename_objectives_unpenalized = filename_prefix + '_objectives_unpenalized.txt'
filename_objectives_penalties = filename_prefix + '_objectives_penalties.txt'
filename_objectives_primal = filename_prefix + '_objectives_primal.txt'
filename_running_time = filename_prefix + '_running_time.txt'
filename_json = filename_prefix + '.json'


objective_vals_list = []
objective_unpenalized_vals_list = []
objective_penalties_list = []
objective_primal_list = []
time_spent_list = []

primal_samples = None
dual_funcs_vals = None

for run_i in range(num_of_runs):
    rng_training = np.random.RandomState(990000 + 10 * run_i)
    rng_sampling_dual = np.random.default_rng(991000 + 10 * run_i)
    rng_sampling_primal_prop = np.random.default_rng(992000 + 10 * run_i)
    rng_sampling_primal_acc = np.random.default_rng(993000 + 10 * run_i)
    t0 = time.time()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        marg_sample_generator = meas_CPWA_generator(rng_training, BATCH_MARGS, grid_densities_list)
        refjoint_sample_generator = meas_CPWA_generator(rng_training, BATCH_REFJOINT, grid_densities_list)

        objective_vals_runtime_list = []
        max_superreplic_diff_runtime_list = []
        rejsamp_accepted_samples = np.zeros([0, NUM_OF_MARGS, DIM])
        for iter in range(NUM_OF_ITERS):
            marg_samples = next(marg_sample_generator)
            refjoint_samples = next(refjoint_sample_generator)

            (_, training_runtime_objective, training_runtime_superreplic_diffs) = sess.run([train_op, objective_func, superreplic_diff_samples], feed_dict={symb_samples_from_margs: marg_samples, symb_samples_from_refjoint: refjoint_samples, global_step: max(min(DECAY_STEPS, iter-DECAY_START), 0)})

            objective_vals_runtime_list.append(training_runtime_objective)
            max_superreplic_diff_runtime_list.append(np.max(training_runtime_superreplic_diffs))

            if iter % 500 == 0:
                current_learning_rate = sess.run(learning_rate, feed_dict={global_step: max(min(DECAY_STEPS, iter-DECAY_START), 0)})
                print('Run ' + str(run_i) + ' current iteration: ' + str(iter))
                print('Run ' + str(run_i) + ' current learning rate: ' + str(current_learning_rate))
                print('Run ' + str(run_i) + ' current dual value: ' + str(np.mean(objective_vals_runtime_list[max(0, iter-2000):])))
        
        # evaluate the dual objective function after training has terminated
        final_dual_sample_generator = meas_CPWA_generator(rng_sampling_dual, NUM_OF_MC_SAMP_FINAL_OBJ, grid_densities_list)
        final_dual_samples = next(final_dual_sample_generator)

        (final_obj, final_obj_unpen, final_penalty) = sess.run([objective_func, sum_of_integrals_MonteCarlo, penalty_MonteCarlo], feed_dict={symb_samples_from_margs: final_dual_samples, symb_samples_from_refjoint: final_dual_samples})

        # only evaluate the primal in the first run
        if run_i == 0 and COMPUTE_PRIMAL:

            # evaluate the primal objective function via rejection sampling
            final_primal_sample_generator = meas_CPWA_generator(rng_sampling_primal_prop, NUM_OF_MC_SAMP_FINAL_OBJ, grid_densities_list)
            final_primal_samples = np.zeros([0, NUM_OF_MARGS, DIM])

            # use the maximum density in the last 1000 iterations of training plus a 10% margin as an upper bound for the maximum density
            rejsamp_max_density = max(max_superreplic_diff_runtime_list[-1000:]) * 1.1

            while True:
                rejsamp_proposed_samples = next(final_primal_sample_generator)
                rejsamp_acceptence_rv_samples = rng_sampling_primal_acc.random([NUM_OF_MC_SAMP_FINAL_OBJ])

                rejsamp_samples_superreplic_diffs = sess.run(superreplic_diff_samples, feed_dict={symb_samples_from_margs: rejsamp_proposed_samples, symb_samples_from_refjoint: rejsamp_proposed_samples})

                if PENALTY_TYPE == 'L2':
                    rejsamp_proposed_samples_densities = rejsamp_samples_superreplic_diffs
                elif PENALTY_TYPE == 'exp':
                    rejsamp_proposed_samples_densities = np.exp(PENALTY_PARAM_GAMMA * rejsamp_samples_superreplic_diffs)

                new_primal_samples = rejsamp_proposed_samples[rejsamp_acceptence_rv_samples * rejsamp_max_density <= rejsamp_proposed_samples_densities, :, :]
                final_primal_samples = np.append(final_primal_samples, new_primal_samples, axis=0)

                print('Run ' + str(run_i) + ': ' + str(final_primal_samples.shape[0]) + ' samples accepted')

                if final_primal_samples.shape[0] >= NUM_OF_MC_SAMP_FINAL_OBJ:
                    final_primal_samples = final_primal_samples[0:NUM_OF_MC_SAMP_FINAL_OBJ, :, :]
                    break
            
            final_obj_primal = np.mean(costfunc_np(final_primal_samples))
        else:
            final_primal_samples = None
            final_obj_primal = np.nan

        if run_i == 0:
            # evaluate the neural networks at the input grid points
            dual_funcs_vals = sess.run(potential_marg_samples, feed_dict={symb_samples_from_margs: np.tile(np.expand_dims(dual_funcs_eval_points, axis=1), [1, NUM_OF_MARGS, 1])})
            primal_samples = final_primal_samples

        print('Finished')
        print('Run ' + str(run_i) + ': final approx. real objective: ' + str(-final_obj_unpen))
        print('Run ' + str(run_i) + ': final approx. penalty: ' + str(final_penalty))
        print('Run ' + str(run_i) + ': final approx. entropic objective: ' + str(-final_obj))
        print('Run ' + str(run_i) + ': final approx. primal objective: ' + str(-final_obj_primal))
        print('\n')
    
    time_spent_list.append(time.time()-t0)
    objective_vals_list.append(-final_obj)
    objective_unpenalized_vals_list.append(-final_obj_unpen)
    objective_penalties_list.append(final_penalty)
    objective_primal_list.append(-final_obj_primal)

    output_dict = {
        "objective_vals_list": objective_vals_list,
        "objective_unpenalized_vals_list": objective_unpenalized_vals_list,
        "objective_penalties_list": objective_penalties_list,
        "objective_primal_vals_list": objective_primal_list,
        "time_spent_list": time_spent_list,
        "primal_samples": np.squeeze(primal_samples).tolist(),
        "dual_funcs_eval_points": np.squeeze(dual_funcs_eval_points).tolist(),
        "dual_funcs_vals": (-np.squeeze(dual_funcs_vals)).tolist()
    }

    print('\n')
    print(objective_vals_list)
    print(objective_unpenalized_vals_list)
    print(objective_penalties_list)
    print(objective_primal_list)
    print(time_spent_list)
    print('\n')
    np.savetxt(filename_objectives, objective_vals_list)
    np.savetxt(filename_objectives_unpenalized, objective_unpenalized_vals_list)
    np.savetxt(filename_objectives_penalties, objective_penalties_list)
    np.savetxt(filename_objectives_primal, objective_primal_list)
    np.savetxt(filename_running_time, time_spent_list)

    with open(filename_json, 'w') as json_file:
        json.dump(output_dict, json_file)

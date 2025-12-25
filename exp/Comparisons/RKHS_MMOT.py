import numpy as np

# Cleaned up version of the code

def RKHS_MMOT(marg_samps_generator, cost_func, num_of_margs, dim, kernel_func, penalty_param_gamma, num_of_iters, decay_scheme='poly', decay_update_frequency = 10000, decay_poly_exponent=2, learning_rate_begin=0.0005, learning_rate_final=10**(-7), penalty_type='exp', num_of_MC_samp_final_obj = 50000, dual_funcs_eval_points=None, log_tag = ''):
    """
    calculates a multi-marginal OT problem by a reproducing kernel hilbert space approach
    We follow algorithm 3 in Genevay, Cuturi, Peyré, Bach "Stochastic Optimization for Large-scale OT"
    :param marg_samps_generator: marginals generator functions.
    Should take no inputs and generate [k, d] marginals
    :param cost_func: function in OT objective. Evaluated at [k, d] shape entries
    :param num_of_margs: number of marginals
    :param dim: dimension of marginals. Should be the same for each marginal
    :param kernel_func: kernel basis, should take two np array inputs of shape [n, k, d], [n, k, d]
    and return [n, k] shape
    :param penalty_param_gamma: factor for penalization
    :param num_of_iters: number of iterations of training
    :param decay_scheme: how step size of SGD declines ('poly' for polynomial; 'exp' for geometric; 'inv' for harmonic)
    :param decay_update_frequency: how frequent the learning rate is updated
    :param decay_poly_exponent: if decay_scheme='poly', this is the exponent of the polynomial decay
    :param learning_rate_begin: initial step size
    :param learning_rate_final: final step size (if polynomial decay)
    :param penalty_type: type of penalty function
    :param num_of_MC_samp_final_obj: number of Monte Carlo samples to generate in order to approximate the final objective value
    :param dual_funcs_eval_points: points at which the final dual functions are evaluated, should take a list of np arrays with [n, d] shape
    :param log_tag: string prepended to log lines
    :return: objective with penalty, objective without penalty, penalty, optimizer, dual function values
    """
    # weights in the linear combinations representing the potential functions via RKHS vectors; weights are the same for all marginals
    runtime_RKHS_weights_list = np.zeros([num_of_iters, 1]) 

    # points with respect to which the RKHS vectors are constructed
    runtime_RKHS_points_list = np.zeros([num_of_iters, num_of_margs, dim])

    # values of the potential functions evaluated at the newly sampled point 
    runtime_RKHS_evals_list = np.zeros([num_of_iters, num_of_margs]) 
    
    runtime_penalty_list = np.zeros(num_of_iters)
    samps_generator = marg_samps_generator()
    sample = next(samps_generator)
    runtime_RKHS_points_list[0, :, :] = sample

    if penalty_type == 'exp':
        def penalty_func(x):
            return np.exp(x-1)
        
        def penalty_derivative_func(x):
            return np.exp(x-1)
    elif penalty_type == 'L2':
        def penalty_func(x):
            return np.maximum(x, 0) ** 2
        
        def penalty_derivative_func(x):
            return 2*np.maximum(x, 0)

    learning_rate = learning_rate_begin
    runtime_RKHS_weights_list[0] = learning_rate * (1 - penalty_derivative_func(-penalty_param_gamma * cost_func(sample))) # since u[0] = 0
    penalty_param_inv_gamma = 1/penalty_param_gamma
    runtime_penalty_list[0] = penalty_param_inv_gamma * penalty_func(-penalty_param_gamma * cost_func(sample))

    if decay_scheme == 'exp':
        exp_decay_rate = (learning_rate_final/learning_rate_begin) ** (decay_update_frequency/num_of_iters)

    for iter in range(1, num_of_iters):
        sample = next(samps_generator)
        runtime_RKHS_points_list[iter, :, :] = sample
        runtime_RKHS_evals_list[iter, :] = np.sum(np.tile(runtime_RKHS_weights_list[:iter], [1, num_of_margs]) * kernel_func(np.tile(sample, [iter, 1, 1]), runtime_RKHS_points_list[:iter, :, :]), axis=0)
        subreplic_diff = np.sum(runtime_RKHS_evals_list[iter, :]) - cost_func(sample)
        runtime_RKHS_weights_list[iter] = learning_rate * (1 - penalty_derivative_func(penalty_param_gamma * subreplic_diff))
        runtime_penalty_list[iter] = penalty_param_inv_gamma * penalty_func(penalty_param_gamma * subreplic_diff)

        if iter % decay_update_frequency == 0:
            if decay_scheme == 'poly':
                learning_rate = (learning_rate_begin - learning_rate_final) * (1 - iter / num_of_iters) ** decay_poly_exponent + learning_rate_final
            elif decay_scheme == 'exp':
                learning_rate *= exp_decay_rate
            elif decay_scheme == 'inv':
                learning_rate = learning_rate_begin / (iter / decay_update_frequency)

        if iter % 5000 == 0:
            print(iter)
            print('Current learning rate: ' + str(learning_rate))

            # approximate the current objective and penalty by the past 5000 iterations
            current_objective_approx = np.mean(np.sum(runtime_RKHS_evals_list[iter-5000:iter, :], axis=1))
            current_penalty_approx = np.mean(runtime_penalty_list[iter-5000:iter])
            print(log_tag + ': current approx. real objective: ' + str(current_objective_approx))
            print(log_tag + ': current approx. penalty: ' + str(current_penalty_approx))
            print(log_tag + ': current approx. entropic objective: ' + str(current_objective_approx - current_penalty_approx))

    # after all the iterations, approximate the final objective and penalty by Monte Carlo integration
    final_obj = 0
    final_penalty = 0

    for samp_i in range(num_of_MC_samp_final_obj):
        sample = next(samps_generator)
        final_RKHS_evals_sum = np.sum(np.tile(runtime_RKHS_weights_list, [1, num_of_margs]) * kernel_func(np.tile(sample, [num_of_iters, 1, 1]), runtime_RKHS_points_list))
        final_obj += final_RKHS_evals_sum / num_of_MC_samp_final_obj
        subreplic_diff = final_RKHS_evals_sum - cost_func(sample)
        final_penalty += penalty_param_inv_gamma * penalty_func(penalty_param_gamma * subreplic_diff) / num_of_MC_samp_final_obj

        if samp_i % 5000 == 0:
            print('Sampling progress: ' + str(samp_i))
    

    if dual_funcs_eval_points is not None:
        dual_funcs_vals_list = []

        for marg_i in range(num_of_margs):
            dual_funcs_inputs = dual_funcs_eval_points[marg_i]
            num_of_dual_funcs_inputs = dual_funcs_inputs.shape[0]
            dual_funcs_vals = []

            for input_i in range(num_of_dual_funcs_inputs):
                dual_func_val = np.sum(runtime_RKHS_weights_list * kernel_func(np.tile(dual_funcs_inputs[input_i,:], [num_of_iters, 1, 1]), runtime_RKHS_points_list[:, [marg_i], :])).tolist()
                dual_funcs_vals.append(dual_func_val)
            
            dual_funcs_vals_list.append(dual_funcs_vals)
    else:
        dual_funcs_vals_list = None

    
    print('Finished')
    print(log_tag + ': final approx. real objective: ' + str(final_obj))
    print(log_tag + ': final approx. penalty: ' + str(final_penalty))
    print(log_tag + ': final approx. entropic objective: ' + str(final_obj - final_penalty))
    print('\n')

    return final_obj - final_penalty, final_obj, final_penalty, [runtime_RKHS_points_list, runtime_RKHS_weights_list], dual_funcs_vals_list


def RKHS_Gauss_kernel_func(x, y, sig=2):
    out = (x-y) ** 2
    out = np.sum(out, 2)
    return np.exp(-out/(2 * sig ** 2))


def RKHS_Laplace_kernel_func(x, y, sig=2):
    out = (x-y) ** 2
    out = np.sum(out, 2)
    out = np.sqrt(out)
    return np.exp(-out / sig)
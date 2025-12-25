import numpy as np
import json

if __name__ == "__main__":
    import matplotlib.pyplot as plt


def eval_CPWA_func(inputs, grid_vals):
    """
    Evaluate a two-dimensional continuous piece-wise affine function on [0,3] * [0,3] with respect to a triangular mesh
    :param inputs: numpy array with shape [n, 2] containing the input points
    :param grid_vals: numpy array with shape [4, 4] containing the function values at the grid points
    :return vals: numpy array with shape [n] containing the evaluated function values
    """
    num_of_inputs = inputs.shape[0]
    vals = np.zeros([num_of_inputs])
    inputs_sum = np.sum(inputs, axis=1)

    for i_x in range(3):
        for i_y in range(3):
            # lower triangular half of the grid
            vals += ((inputs[:, 0] - i_x) * (grid_vals[i_x + 1, i_y] - grid_vals[i_x, i_y]) \
                     + (inputs[:, 1] - i_y) * (grid_vals[i_x, i_y + 1] - grid_vals[i_x, i_y]) + grid_vals[i_x, i_y]) \
                        * ((inputs[:, 0] > i_x) & (inputs[:, 1] > i_y) & (inputs_sum < i_x + i_y + 1))

            # lower triangular half of the grid
            vals += ((i_x + 1 - inputs[:, 0]) * (grid_vals[i_x, i_y + 1] - grid_vals[i_x + 1, i_y + 1]) \
                     + (i_y + 1 - inputs[:, 1]) * (grid_vals[i_x + 1, i_y] - grid_vals[i_x + 1, i_y + 1]) + grid_vals[i_x + 1, i_y + 1]) \
                        * ((inputs[:, 0] < i_x + 1) & (inputs[:, 1] < i_y + 1) & (inputs_sum > i_x + i_y + 1))
    return vals


def meas_CPWA_generator(rng, batch_size, grid_densities_list):
    """
    Generate samples from a two-dimensional probability measure with continuous piece-wise affine density on [0,3] * [0,3] with respect to a triangular mesh
    :param rng: RandomState object
    :param batch_size: number of samples to be generated in each mini-batch
    :param grid_densities_list: list of numpy arrays each with shape [4, 4] representing the unnormalized density functions at the grid points
    """

    num_of_margs = len(grid_densities_list)
    
    # generate 10 times of batch_size of samples at once before acceptance/rejection
    rejsamp_size = batch_size * 10

    max_densities = []

    # compute the maximum unnormalized density of each marginal
    for marg_i in range(num_of_margs):
        max_densities.append(np.max(grid_densities_list[marg_i]))

    while True:
        samps = np.zeros([batch_size, num_of_margs, 2])
        for marg_i in range(num_of_margs):
            marg_samps = np.zeros([0, 2])

            while True:
                proposed_samps = rng.random([rejsamp_size, 2]) * 3
                accept_probs = eval_CPWA_func(proposed_samps, grid_densities_list[marg_i]) / max_densities[marg_i]
                acceptance = rng.random([rejsamp_size])
                marg_samps = np.append(marg_samps, proposed_samps[acceptance <= accept_probs], axis=0)

                if marg_samps.shape[0] >= batch_size:
                    samps[:, marg_i, :] = marg_samps[0:batch_size, :]
                    break
        
        yield samps


def main():
    # In the main function, we plot the two-dimensional histograms of a large sample from the marginals
    rng = np.random.RandomState(1000)

    with open('WBMM_densities.json', 'r') as jsonfile:
        WBMM_densities = json.load(jsonfile)

    num_of_margs = len(WBMM_densities)

    grid_densities_list = []

    for marg_i in range(num_of_margs):
        marg_densities = np.reshape(np.array(WBMM_densities[marg_i]), shape=[4,-1])
        grid_densities_list.append(marg_densities)
    
    generator = meas_CPWA_generator(rng, batch_size=1000000, grid_densities_list=grid_densities_list)
    samps = next(generator)

    fig, axs = plt.subplots(ncols=5, figsize=(15, 3))

    for marg_i in range(len(axs.flat)):
        axs.flat[marg_i].set_title('Marginal ' + str(marg_i + 1))
        axs.flat[marg_i].hist2d(samps[:, marg_i, 0], samps[:, marg_i, 1], bins=50)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
load('exp/exp_inputs.mat');
load('exp/exp_invcdf.mat');
load('exp/exp_rst_LB.mat', 'coup_meas_cell');

% compute the inverse cdf of the marginal distributions
marginv = @(i, u)(interp1(mixtrnorm_invcdf_cell{i, 1}, ...
        mixtrnorm_invcdf_cell{i, 2}, u));

% Monte Carlo sample numbers and repetition numbers
samp_no = 1e6;
rep_no = 1000;
batch_size = 1e5;

MMOT_UB_samps_cell = cell(step_no, 1);
MMOT_UB_errbar = zeros(step_no, 2);

for step_id = 1:step_no
    if isempty(coup_meas_cell{step_id})
        continue;
    end
    
    rand_stream = RandStream('combRecursive', 'Seed', 1000 + step_id);
    
    fprintf('Setting %d:\n', step_id);
    MMOT_UB_samps_cell{step_id} = MMOT_CPWA_primal_approx(CPWA, ...
        coup_meas_cell{step_id}.atoms, coup_meas_cell{step_id}.probs, ...
        marginv, samp_no, rep_no, batch_size, rand_stream, [], true);
    MMOT_UB_errbar(step_id, :) = quantile( ...
        MMOT_UB_samps_cell{step_id}, [0.025, 0.975]);
    
    save('exp/exp_rst_UB.mat', 'MMOT_UB_samps_cell', 'MMOT_UB_errbar', ...
        'samp_no', 'rep_no', 'batch_size');
end
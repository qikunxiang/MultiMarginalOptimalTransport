% Run step 2: computing the upper bounds by first computing reassemblies
% and then approximating the integrals with respect to the reassemblies via
% Monte Carlo integration

load('exp/inputs.mat');
load('exp/invcdf.mat');
load('exp/rst_LB.mat', 'coup_meas_cell');

% compute the inverse cdf of the marginal distributions
marginv = @(i, u)(interp1(mixnorm_invcdf_cell{i, 1}, ...
        mixnorm_invcdf_cell{i, 2}, u));

% Monte Carlo sample numbers and repetition numbers
sampno = 1e6;
repno = 1000;
batchsize = 1e5;

MMOT_UB_samps_cell = cell(stepno, 1);
MMOT_UB_errbar = zeros(stepno, 2);

for stepid = 1:stepno
    if isempty(coup_meas_cell{stepid})
        continue;
    end
    
    rng(1000 + stepid, 'combRecursive');
    
    MMOT_UB_samps_cell{stepid} = MMOT_CPWA_primal_approx(CPWA, ...
        coup_meas_cell{stepid}.atoms, coup_meas_cell{stepid}.probs, ...
        marginv, sampno, repno, batchsize, true);
    MMOT_UB_errbar(stepid, :) = quantile( ...
        MMOT_UB_samps_cell{stepid}, [0.025, 0.975]);
    
    save('exp/rst_UB.mat', ...
        'MMOT_UB_samps_cell', 'MMOT_UB_errbar', ...
        'sampno', 'repno', 'batchsize');
end
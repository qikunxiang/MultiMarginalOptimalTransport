% Example of multi-marginal optimal transport (MMOT) problem with
% 1D marginals and a continuous piece-wise affine cost function
CONFIG = CPWAMM_config();

marg_num = 100;

marg_support_range = [-10, 10];
marg_comp_num_range = [3, 5];

% parameters of the marginals
marg_params_cell = cell(marg_num, 1);

rng(1000, "combRecursive");

for marg_id = 1:marg_num
    mixnorm_comp_num = randi(marg_comp_num_range, 1, 1);
    mixnorm_weights = ones(mixnorm_comp_num, 1) / mixnorm_comp_num;
    mixnorm_mean_list = randn(mixnorm_comp_num, 1) * 3;
    mixnorm_std_list = sqrt(1 ./ gamrnd(3, 0.1, mixnorm_comp_num, 1));

    marg_params_cell{marg_id} = {marg_support_range(1); ...
        marg_support_range(2); struct('weights', mixnorm_weights, ...
        'mean_list', mixnorm_mean_list, 'std_list', mixnorm_std_list)};
end

% cost function
% f(x) = |dir1' * x - thres1| + |dir2' * x - thres2| ...
%        - |dir3' * x - thres3| - |dir4' * x - thres4|
cost_plus_num = 2;
cost_minus_num = 2;

rng(1500, "combRecursive");

weights_plus = randn(cost_plus_num, marg_num);
weights_plus = weights_plus ./ sqrt(sum(weights_plus .^ 2, 2));
weights_minus = randn(cost_minus_num, marg_num);
weights_minus = weights_minus ./ sqrt(sum(weights_minus .^ 2, 2));
intercepts_plus = randn(cost_plus_num, 1) * 2;
intercepts_minus = randn(cost_minus_num, 1) * 2;

costfunc = struct;
costfunc.plus = cell(cost_plus_num, 2);
for plus_id = 1:cost_plus_num
    costfunc.plus{plus_id, 1} = repmat(weights_plus(plus_id, :), ...
        2, 1) .* [1; -1];
    costfunc.plus{plus_id, 2} = [-1; 1] * intercepts_plus(plus_id);
end

costfunc.minus = cell(cost_minus_num, 2);
for minus_id = 1:cost_minus_num
    costfunc.minus{minus_id, 1} = repmat(weights_minus(minus_id, :), ...
        2, 1) .* [1; -1];
    costfunc.minus{minus_id, 2} = [-1; 1] * intercepts_minus(minus_id);
end

% test functions for the marginals
knot_num_list = [4; 8; 16; 32]' + 1;
test_num = length(knot_num_list);

testfuncs_cell = cell(test_num, 1);

for test_id = 1:test_num
    testfuncs_cell{test_id} = cell(marg_num, 1);

    for marg_id = 1:marg_num
        % apply the inverse CDF transform to get appropriately placed knots
        knot_probs = linspace(0, 1, knot_num_list(test_id))';
        meas = ProbMeas1DMixNorm(marg_params_cell{marg_id}{:});
        testfuncs_knots = meas.evaluateInverseCDF(knot_probs);

        testfuncs_cell{test_id}{marg_id} = {testfuncs_knots};
    end
end

save(CONFIG.SAVEPATH_INPUTS, ...
    'marg_num', ...
    'marg_params_cell', ...
    'costfunc', ...
    'test_num', ...
    'testfuncs_cell');
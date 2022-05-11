rng(1000, 'combRecursive');

N = 50;

% problem specification

% the mu and sigma^2 parameters of the mixture of truncated normal
% marginals 
mixtrnorm_mu_cell = cell(N, 1);
mixtrnorm_sig2_cell = cell(N, 1);
mixtrnorm_w_cell = cell(N, 1);
% truncation limits 
mixtrnorm_trunc = [-10; 10];
mixtrnorm_compno = 3;

for i = 1:N
    mixtrnorm_mu_cell{i} = randn(mixtrnorm_compno, 1) * 3;
    mixtrnorm_sig2_cell{i} = 1 ./ gamrnd(3, 1, mixtrnorm_compno, 1);
    mixtrnorm_w_cell{i} = ones(mixtrnorm_compno, 1) / mixtrnorm_compno;
end

% f(x) = |dir1' * x - thres1| + |dir2' * x - thres2| ...
%        - |dir3' * x - thres3| - |dir4' * x - thres4|
K_plus = 2;
K_minus = 2;

dir_plus = randn(K_plus, N);
dir_plus = dir_plus ./ sqrt(sum(dir_plus .^ 2, 2));
dir_minus = randn(K_minus, N);
dir_minus = dir_minus ./ sqrt(sum(dir_minus .^ 2, 2));
trunc_plus = randn(K_plus, 1) * 2;
trunc_minus = randn(K_minus, 1) * 2;

CPWA = struct;
CPWA.plus = cell(K_plus, 2);
for funcid = 1:K_plus
    CPWA.plus{funcid, 1} = repmat(dir_plus(funcid, :), 2, 1) .* [1; -1];
    CPWA.plus{funcid, 2} = [1; -1] * trunc_plus(funcid);
end

CPWA.minus = cell(K_minus, 2);
for funcid = 1:K_plus
    CPWA.minus{funcid, 1} = repmat(dir_minus(funcid, :), 2, 1) .* [1; -1];
    CPWA.minus{funcid, 2} = [1; -1] * trunc_minus(funcid);
end

% an upper bound on the Lipschitz constant of the CPWA function in each
% dimension
CPWA_Lip_bd = sum(abs(dir_plus), 1)' + sum(abs(dir_minus), 1)';

% the tolerance value used when solving LSIP
LSIP_tolerance = 1e-4;

% approximation scheme
knot_no_list = round(exp(linspace(log(5), log(100), 8)))';
step_no = length(knot_no_list);

knots_cell = cell(step_no, 1);
expval_cell = cell(step_no, 1);
bd_cell = cell(step_no, 1);
err_bound_list = zeros(step_no, 1);
coef_num_cell = cell(step_no, 1);

knots_hist_cell = cell(N, 1);
bd_hist_cell = cell(N, 1);

for i = 1:N
    % first generate the maximum number of knots
    [~, knots_hist_cell{i}, bd_hist_cell{i}] ...
        = mixtruncnorm_momentset_greedy( ...
        mixtrnorm_mu_cell{i}, mixtrnorm_sig2_cell{i}, ...
        mixtrnorm_w_cell{i}, mixtrnorm_trunc, knot_no_list(end));
end

for step_id = 1:step_no
    knots = cell(N, 1);
    expval = cell(N, 1);
    bd_list = zeros(N, 1);
    coef_num_list = zeros(N, 1);
    
    for i = 1:N
        % retrive a subset of knots
        knots{i} = sort(knots_hist_cell{i}(1:knot_no_list(step_id)), ...
            'ascend');

        % the corresponding upper bound on the Wasserstein-1 radius
        bd_list(i) = bd_hist_cell{i}(knot_no_list(step_id));
        
        % compute the corresponding integrals
        expval{i} = mixtruncnorm_momentset(mixtrnorm_mu_cell{i}, ...
        mixtrnorm_sig2_cell{i}, mixtrnorm_w_cell{i}, knots{i});

        % the number of coefficients (with the constant intercept)
        coef_num_list(i) = length(knots{i});
    end
    
    knots_cell{step_id} = knots;
    expval_cell{step_id} = expval;
    bd_cell{step_id} = bd_list;
    err_bound_list(step_id) = LSIP_tolerance + bd_list' * CPWA_Lip_bd;
    coef_num_cell{step_id} = coef_num_list;
end

marg1 = cell(N, 2);

for i = 1:N
    % generate a discrete measure in the moment set
    [marg1{i, 1}, marg1{i, 2}] = momentset1d_measinit_bounded( ...
        knots_cell{1}{i}, expval_cell{1}{i});
end

% compute the comonotone coupling as the initial measure
[joint_atoms, joint_prob] = comonotone_coupling(marg1);

coup_init1 = joint_atoms;

save('exp/exp_inputs.mat', ...
    'N', 'mixtrnorm_mu_cell', 'mixtrnorm_sig2_cell', ...
    'mixtrnorm_w_cell', 'mixtrnorm_trunc', 'CPWA', 'CPWA_Lip_bd', ...
    'LSIP_tolerance', 'knot_no_list', 'step_no', 'knots_cell', ...
    'expval_cell', 'bd_cell', 'err_bound_list', 'coef_num_cell', ...
    'coup_init1', '-v7.3');
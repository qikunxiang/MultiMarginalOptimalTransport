function [LSIP_UB, LSIP_LB, coup_meas, dual_func, output, options] ...
    = MMOT_CPWA_cutplane(CPWA, knots, expval, coup_init, options)
% Compute MMOT with CPWA cost function using the cutting plane algorithm
% Inputs:
%       CPWA: the specification of the CPWA cost function
%       knots: the knots in each dimension where the first knot and the
%       last knot in each dimension indicates the support of the
%       distribution 
%       expval: the expected values of the CPWA basis functions in each
%       dimension
%       coup_init: the initial coupling where each row represents an atom
%       (weights are omitted)
%       options: struct with the following fields:
%           tolerance: numerical tolerance (default is 1e-4)
%           reduce_cuts: struct containing options related to reducing cuts
%               thres: constraints (cuts) with slackness above the
%               threshold will be removed (default is inf)
%               freq: frequency of reducing cuts (default is 20)
%               max_iter: no cut will be performed beyond this iteration
%               (default is inf)
%           display: whether to display output (default is true)
%           log_file: path to the log file where the outputs will be
%           written (default is '')
%           LP_params: additional parameters for the LP solver
%           global_params: additional parameters for the global
%           optimization solver
%           rescue: struct containing parameters for saving and loading
%           save files 
%               save_file: path to save the progress (default is '')
%               save_interval: interval to save progress (default is 100)
%               load_file: path to load a save file before running the
%               algorithm (default is '')
% Outputs:
%       LSIP_UB: the computed upper bound for the LSIP problem
%       LSIP_LB: the computed lower bound for the LSIP problem
%       coup_meas: the optimized coupling with fields atoms and probs
%       dual_func: struct representing the dual optimizer
%           y0: the constant intercept
%           y_cell: cell array containing the coefficients of interpolation
%           functions for each measure
%       output: struct containing additional outputs
%           total_time: total time spent in the algorithm
%           iter: number of iterations
%           LP_time: time spent solving LP problems
%           global_time: time spent solving global optimization problems
%       options: return the options in order to retrive some default values

% start the timer
total_timer = tic;

if ~exist('options', 'var') || isempty(options)
    options = struct;
end

if ~isfield(options, 'tolerance') || isempty(options.tolerance)
    options.tolerance = 1e-4;
end

if ~isfield(options, 'reduce_cuts') || isempty(options.reduce_cuts)
    options.reduce_cuts = struct;
end

if ~isfield(options.reduce_cuts, 'thres') ...
        || isempty(options.reduce_cuts.thres)
    options.reduce_cuts.thres = inf;
end

if ~isfield(options.reduce_cuts, 'freq') ...
        || isempty(options.reduce_cuts.freq)
    options.reduce_cuts.freq = 20;
end

if ~isfield(options.reduce_cuts, 'max_iter') ...
        || isempty(options.reduce_cuts.max_iter)
    options.reduce_cuts.max_iter = inf;
end

if ~isfield(options, 'display') || isempty(options.display)
    options.display = true;
end

if ~isfield(options, 'log_file') || isempty(options.log_file)
    options.log_file = '';
end

if ~isfield(options, 'rescue') || isempty(options.rescue)
    options.rescue = struct;
end

if ~isfield(options.rescue, 'save_file') ...
        || isempty(options.rescue.save_file)
    options.rescue.save_file = '';
end

if ~isfield(options.rescue, 'save_interval') ...
        || isempty(options.rescue.save_interval)
    options.rescue.save_interval = 100;
end

if ~isfield(options.rescue, 'load_file') ...
        || isempty(options.rescue.load_file)
    options.rescue.load_file = '';
end

% number of input measures
N = length(knots);
assert(length(expval) == N, 'expected values invalid');

% check the integrity of the CPWA function specification
if isfield(CPWA, 'plus')
    for i = 1:size(CPWA.plus, 1)
        assert(size(CPWA.plus{i, 1}, 1) == length(CPWA.plus{i, 2}), ...
            'CPWA plus part mis-specified');
        assert(size(CPWA.plus{i, 1}, 2) == N, ...
            'CPWA plus part mis-specified');
    end
end

if isfield(CPWA, 'minus')
    for i = 1:size(CPWA.minus, 1)
        assert(size(CPWA.minus{i, 1}, 1) == length(CPWA.minus{i, 2}), ...
            'CPWA minus part mis-specified');
        assert(size(CPWA.minus{i, 1}, 2) == N, ...
            'CPWA minus part mis-specified');
    end
end

% number of knots in each dimension
knot_no_list = zeros(N, 1);

% the objective vectors where the first basis function is removed for
% identification
v_cell = cell(N, 1);

% the dual optimizer
dual_func = struct;
dual_func.y_cell = cell(N, 1);

for i = 1:N
    knot_no_list(i) = length(knots{i});
    
    if i == 1
        v_cell{i} = expval{i};
        dual_func.y_cell{i} = zeros(knot_no_list(i), 1);
    else
        % the first basis function is removed for identification
        v_cell{i} = expval{i}(2:end);
        dual_func.y_cell{i} = zeros(knot_no_list(i) - 1, 1);
    end
    
    assert(size(expval{i}, 1) == knot_no_list(i), ...
        'knots and expected values mismatch');
end

% the offset of points in each triangulation
var_offset = [0; cumsum(knot_no_list(1:end - 1) - 1) + 1];

% total number of decision variables in the LP problem including y0
LP_var_len = sum(knot_no_list - 1) + 1;

% assemble the Gurobi model for the linear programming problem
LP_model = struct;
LP_model.modelsense = 'max';
LP_model.obj = vertcat(v_cell{:});
LP_model.sense = '<';
LP_model.lb = -inf(LP_var_len, 1);
LP_model.ub = inf(LP_var_len, 1);

% generate the initial constraints coming from the initial coupling
[A_init, rhs_init] = MMOT_CPWA_feascons(coup_init, knots, CPWA);
LP_model.A = A_init;
LP_model.rhs = rhs_init;

% parameters of the LP solver
LP_params = struct;

% disable output by default
LP_params.OutputFlag = 0;

% since the gurobi MATLAB interface does not support callback, set a
% 30-minute time limit on the LP solver; if the solver does not converge
% when the time limit is hit, it assumes that numerical issues have
% occurred and restarts the solution process with NumericFocus = 3 and
% LPWarmStart = 2 and TimeLimit = inf
LP_params.TimeLimit = 1800;

% set the additional parameters for the LP solver
if isfield(options, 'LP_params') && ~isempty(options.LP_params)
    LP_params_fields = fieldnames(options.LP_params);
    LP_params_values = struct2cell(options.LP_params);
    
    for fid = 1:length(LP_params_fields)
        LP_params.(LP_params_fields{fid}) = LP_params_values{fid};
    end
end

% parameters of the global (MILP) solver
global_params = struct;
global_params.OutputFlag = 0;
global_params.IntFeasTol = 1e-6;
global_params.FeasibilityTol = 1e-8;
global_params.OptimalityTol = 1e-8;
global_params.PoolSolutions = 100;
global_params.PoolGap = 0.8;
global_params.NodefileStart = 2;
global_params.BestBdStop = 1e-6;
global_params.BestObjStop = -inf;
global_params.MIPGap = 1e-4;
global_params.MIPGapAbs = 1e-10;

% set the additional parameters for the global (MILP) solver
if isfield(options, 'global_params') ...
        && ~isempty(options.global_params)
    global_params_fields = fieldnames(options.global_params);
    global_params_values = struct2cell(options.global_params);
    
    for fid = 1:length(global_params_fields)
        global_params.(global_params_fields{fid}) ...
            = global_params_values{fid};
    end
end

% initialize the lower and upper bounds
LSIP_UB = inf;
LSIP_LB = -inf;
iter = 0;

% initialize the set of constraints (atoms)
atoms_agg = coup_init;

% the basis used for warm start
cbasis = [];
vbasis = [];

% some statistics
LP_time = 0;
global_time = 0;
total_time = 0;

% open the log file
if ~isempty(options.log_file)
    log_file = fopen(options.log_file, 'a');
end

if log_file < 0
    error('cannot open log file');
end

if ~isempty(options.rescue.load_file)
    % load all states from the load file
    load(options.rescue.load_file);
end

% loop until the gap between lower and upper bounds is below the tolerance
while LSIP_UB - LSIP_LB > options.tolerance
    % solve the LP problem
    LP_timer = tic;
    LP_trial_num = 0;
    while true
        LP_params_runtime = LP_params;

        if LP_trial_num == 1
            % if the LP solver has failed once (reaching the time limit
            % without converging), retry with high numeric focus and
            % presolve (with warm-start)
            LP_params_runtime.TimeLimit = inf;
            LP_params_runtime.NumericFocus = 3;
            LP_params_runtime.LPWarmStart = 2;
        end

        LP_result = gurobi(LP_model, LP_params_runtime);

        if strcmp(LP_result.status, 'OPTIMAL')
            break;
        else
            LP_trial_num = LP_trial_num + 1;
        end

        if LP_trial_num >= 2
            % if the LP solver fails after the second trial, report error
            error('unexpected error while solving LP');
        end
    end
    LP_time = LP_time + toc(LP_timer);
    
    if isfield(LP_result, 'vbasis') && ~isempty(LP_result.vbasis) ...
            && isfield(LP_result, 'cbasis') && ~isempty(LP_result.cbasis)
        vbasis = LP_result.vbasis;
        cbasis = LP_result.cbasis;
    end
    
    % store the LP dual optimizer
    LP_dual = LP_result.pi;
    
    % update the upper bound
    LSIP_UB = LP_result.objval;
    
    % compute the value of the dual functions (including the first knot in
    % each dimension)
    val_cell = cell(N, 1);
    for i = 1:N
        if i == 1
            dual_func.y_cell{i} = LP_result.x(var_offset(i) ...
                + (1:knot_no_list(i)));
            val_cell{i} = -dual_func.y_cell{i};
        else
            dual_func.y_cell{i} = LP_result.x(var_offset(i) ...
                + (1:knot_no_list(i) - 1));
            val_cell{i} = -[0; dual_func.y_cell{i}];
        end
    end
    
    % solve the global optimization problem

    % flip the plus and the minus parts of the CPWA function since we are
    % formulating it into a minimization problem rather than a maximization
    % problem
    CPWA_flipped = struct;
    CPWA_flipped.plus = CPWA.minus;
    CPWA_flipped.minus = CPWA.plus;
    global_model = CPWA_diff_min_MILP_gurobi(0, ...
        knots, val_cell, CPWA_flipped);
    global_timer = tic;
    global_result = gurobi(global_model, global_params);
    global_time = global_time + toc(global_timer);

    LSIP_LB = max(LSIP_LB, LSIP_UB + global_result.objbound);
    
    % reduce cuts
    if ~isinf(options.reduce_cuts.thres) ...
            && iter > 0 && iter <= options.reduce_cuts.max_iter ...
            && mod(iter, options.reduce_cuts.freq) == 0
        % the list of constraints to be kept
        constr_keep_list = LP_result.slack <= options.reduce_cuts.thres;

        % always keep the the initial constraints
        constr_keep_list(1:length(rhs_init)) = true;
        
        % update all variables
        atoms_agg = atoms_agg(constr_keep_list, :);
        LP_model.A = LP_model.A(constr_keep_list, :);
        LP_model.rhs = LP_model.rhs(constr_keep_list);
        
        if ~isempty(cbasis)
            cbasis = cbasis(constr_keep_list);
        end
    end
    
    % get a set of approximate optimizers of the global optimization
    % problem
    if isfield(global_result, 'pool')
        x_cut = horzcat(global_result.pool.xn);
        x_cut = x_cut(1:N, [global_result.pool.objval] < 0)';
    else
        x_cut = global_result.x(1:N)';
    end

    % truncate the optimizers to the lower and upper bounds to remove small
    % numerical errors
    x_cut = max(min(x_cut, global_model.ub(1:N)'), global_model.lb(1:N)');

    % aggregate all the possible atoms
    atoms_agg = [atoms_agg; x_cut]; %#ok<AGROW>
    
    % generate new constraints
    [A_new, rhs_new] = MMOT_CPWA_feascons(x_cut, knots, CPWA);
    LP_model.A = [LP_model.A; A_new];
    LP_model.rhs = [LP_model.rhs; rhs_new];
    
    if ~isempty(vbasis) && ~isempty(cbasis)
        LP_model.vbasis = vbasis;
        LP_model.cbasis = [cbasis; zeros(length(rhs_new), 1)];
    else
        LP_model = rmfield(LP_model, 'vbasis');
        LP_model = rmfield(LP_model, 'cbasis');
    end
    
    iter = iter + 1;
    
    % display output
    if options.display
        fprintf('iter = %6d, LB = %10.6f, UB = %10.6f\n', iter, ...
            LSIP_LB, LSIP_UB);
    end

    % write log
    if ~isempty(options.log_file)
        fprintf(log_file, 'iter = %6d, LB = %10.6f, UB = %10.6f\n', ...
            iter, LSIP_LB, LSIP_UB);
    end

    % overwrite manual_save to true to save states while debugging
    manual_save = false;

    % save states
    if ~isempty(options.rescue.save_file) ...
            && (mod(iter, options.rescue.save_interval) == 0 ...
            || manual_save)
        save(options.rescue.save_file, 'A_new', 'atoms_agg', 'cbasis', ...
            'dual_func', 'expval', 'global_model', 'global_result', ...
            'global_time', 'iter', 'knot_no_list', 'knots', ...
            'LP_dual', 'LP_model', 'LP_result', 'LP_time', ...
            'LP_var_len', 'LSIP_LB', 'LSIP_UB', 'N', 'rhs_new', ...
            'total_time', 'val_cell', 'var_offset', 'vbasis', 'x_cut');
    end
end

total_time = total_time + toc(total_timer);

if ~isempty(options.log_file)
    fclose(log_file);   
end

% the approxiamte optimal coupling
probs_nonzero = LP_dual > 0;
coup_meas = struct;
coup_meas.atoms = atoms_agg(probs_nonzero, :);
coup_meas.probs = LP_dual(probs_nonzero);

% prepare additional outputs
output = struct;

output.total_time = total_time;
output.iter = iter;
output.LP_time = LP_time;
output.global_time = global_time;

end